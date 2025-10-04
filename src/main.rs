use gtk::prelude::*;
use gtk::{Application, Box as GtkBox, Orientation, Label, ScrolledWindow, gdk, CssProvider, Align, Entry, DropDown, StringList, MenuButton, Popover, Button};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use gtk::glib;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use chrono::Local;
use reqwest;
use serde_json::json;
use tokio;
use std::io::{Write, Read, BufReader, BufRead};
use libadwaita::prelude::*;
use libadwaita::{HeaderBar, PreferencesGroup, ActionRow, ApplicationWindow as AdwApplicationWindow};
use std::collections::VecDeque;
use base64::Engine;

// --- Runtime-configurable Settings ---
struct AppSettings {
    kobold_url: Mutex<String>,
    // Removed transcription_model since KoboldCPP handles this internally
}

impl AppSettings {
    fn new() -> Self {
        Self {
            kobold_url: Mutex::new("http://localhost:5001".to_string()), // Default KoboldCPP port
        }
    }
}

// Audio capture state
struct AudioCapture {
    process: Option<std::process::Child>,
    buffer: VecDeque<u8>,
    last_transcription: Instant,
    last_significant_audio: Instant,
    last_api_call: Instant,
    is_recording: AtomicBool,
    silence_threshold: f32,
    silence_duration: Duration,
    min_api_interval: Duration,
}

impl AudioCapture {
    fn new() -> Self {
        Self {
            process: None,
            buffer: VecDeque::new(),
            last_transcription: Instant::now(),
            last_significant_audio: Instant::now(),
            last_api_call: Instant::now(),
            is_recording: AtomicBool::new(false),
            silence_threshold: 0.01,
            silence_duration: Duration::from_secs(2),
            min_api_interval: Duration::from_secs(3),
        }
    }

    fn calculate_rms_amplitude(&self, audio_data: &[u8]) -> f32 {
        if audio_data.len() < 2 {
            return 0.0;
        }

        let mut sum_squares = 0i64;
        let sample_count = audio_data.len() / 2;

        for chunk in audio_data.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
            sum_squares += (sample * sample) as i64;
        }

        let mean_square = sum_squares as f64 / sample_count as f64;
        let rms = mean_square.sqrt();
        (rms / 32768.0) as f32
    }

    fn start_recording(&mut self, device: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Ok(());
        }

        println!("Starting continuous audio recording from device: {}", device);

        if device.starts_with("app:") {
            let app_id = &device[4..];

            let _null_sink = Command::new("pactl")
                .args(["load-module", "module-null-sink", &format!("sink_name=capture_sink_{}", app_id)])
                .output();

            let _move_app = Command::new("pactl")
                .args(["move-sink-input", app_id, &format!("capture_sink_{}", app_id)])
                .output();

            let monitor_name = format!("capture_sink_{}.monitor", app_id);
            let mut child = Command::new("ffmpeg")
                .args([
                    "-f", "pulse",
                    "-i", &monitor_name,
                    "-f", "wav",
                    "-ar", "16000",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    "pipe:1",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()?;

            self.process = Some(child);
        } else {
            let mut child = Command::new("ffmpeg")
                .args([
                    "-f", "pulse",
                    "-i", device,
                    "-f", "wav",
                    "-ar", "16000",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    "pipe:1",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()?;

            self.process = Some(child);
        }

        self.is_recording.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop_recording(&mut self) {
        self.is_recording.store(false, Ordering::Relaxed);
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.buffer.clear();
    }

    fn read_audio_data(&mut self) -> Option<Vec<u8>> {
        if let Some(ref mut process) = self.process {
            if let Some(ref mut stdout) = process.stdout {
                let mut reader = BufReader::new(stdout);
                let mut temp_buffer = [0u8; 8192];

                match reader.read(&mut temp_buffer) {
                    Ok(bytes_read) if bytes_read > 0 => {
                        self.buffer.extend(&temp_buffer[..bytes_read]);

                        if self.buffer.len() > 16000 {
                            let recent_audio: Vec<u8> = self.buffer.iter().rev().take(16000).rev().cloned().collect();
                            let amplitude = self.calculate_rms_amplitude(&recent_audio);

                            if amplitude > self.silence_threshold {
                                self.last_significant_audio = Instant::now();
                            }
                        }

                        let should_process = self.buffer.len() > 144_000 &&
                            (self.last_transcription.elapsed() > Duration::from_secs(3) ||
                             (self.buffer.len() > 288_000 &&
                              self.last_significant_audio.elapsed() > self.silence_duration));

                        if should_process {
                            let audio_data: Vec<u8> = self.buffer.drain(..).collect();
                            self.last_transcription = Instant::now();
                            return Some(audio_data);
                        }
                    }
                    Ok(_) => {
                        thread::sleep(Duration::from_millis(100));
                    }
                    Err(_) => {
                        self.stop_recording();
                        return None;
                    }
                }
            }
        }
        None
    }
}

fn get_audio_devices() -> Vec<String> {
    println!("Getting available audio devices...");
    let output = Command::new("pactl")
        .args(["list", "short", "sources"])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout
                .lines()
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() >= 2 {
                        Some(parts[1].to_string())
                    } else {
                        None
                    }
                })
                .collect()
        }
        Err(e) => {
            println!("Error getting audio devices: {}", e);
            vec!["default".to_string()]
        }
    }
}

fn get_running_applications() -> Vec<(String, String)> {
    println!("Getting running applications with audio...");
    let output = Command::new("pactl")
        .args(["list", "sink-inputs"])
        .output();

    let mut applications = Vec::new();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut current_id: Option<String> = None;
            let mut current_app: Option<String> = None;

            for line in stdout.lines() {
                let line = line.trim();

                if line.starts_with("Sink Input #") {
                    if let Some(id) = current_id.as_ref() {
                        if let Some(app) = current_app.as_ref() {
                            applications.push((format!("app:{}", id), app.clone()));
                        }
                    }
                    current_id = line.split('#').nth(1).map(|s| s.to_string());
                    current_app = None;
                }

                if line.starts_with("application.name = ") {
                    current_app = line.split(" = ")
                        .nth(1)
                        .map(|s| s.trim_matches('"').to_string());
                }

                if line.starts_with("media.name = ") && current_app.is_none() {
                    current_app = line.split(" = ")
                        .nth(1)
                        .map(|s| s.trim_matches('"').to_string());
                }
            }

            if let Some(id) = current_id {
                if let Some(app) = current_app {
                    applications.push((format!("app:{}", id), app));
                }
            }
        }
        Err(e) => {
            println!("Error getting applications: {}", e);
        }
    }

    applications
}

async fn translate_with_kobold(audio_data: Vec<u8>, settings: &Arc<AppSettings>) -> Option<String> {
    // Create WAV header for the raw PCM data
    let mut wav_data = Vec::new();

    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&(36u32 + audio_data.len() as u32).to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes());
    wav_data.extend_from_slice(&1u16.to_le_bytes());
    wav_data.extend_from_slice(&1u16.to_le_bytes());
    wav_data.extend_from_slice(&16000u32.to_le_bytes());
    wav_data.extend_from_slice(&32000u32.to_le_bytes());
    wav_data.extend_from_slice(&2u16.to_le_bytes());
    wav_data.extend_from_slice(&16u16.to_le_bytes());
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(audio_data.len() as u32).to_le_bytes());
    wav_data.extend_from_slice(&audio_data);

    // Encode WAV data to base64
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&wav_data);

    let client = reqwest::Client::new();
    let url = format!("{}/api/extra/transcribe", settings.kobold_url.lock().unwrap());

    let request_body = json!({
        "prompt": "",
        "suppress_non_speech": false,
        "langcode": "en", // Always translate to English
        "audio_data": base64_data
    });

    let response = client.post(&url)
        .json(&request_body)
        .header("Content-Type", "application/json")
        .send()
        .await
        .ok()?;

    if response.status().is_success() {
        let json: serde_json::Value = response.json().await.ok()?;
        json.get("text").and_then(|v| v.as_str()).map(|s| s.trim().to_string())
    } else {
        println!("KoboldCPP translation HTTP error: {} - Response: {:?}", response.status(), response.text().await.ok());
        None
    }
}

fn main() {
    let app = Application::builder()
        .application_id("com.toast.scriptinator")
        .build();

    app.connect_activate(|app| {
        let running = Arc::new(AtomicBool::new(false));
        let settings = Arc::new(AppSettings::new());
        let devices = get_audio_devices();
        let applications = get_running_applications();
        let main_vbox = GtkBox::new(Orientation::Vertical, 0);

        let header_bar = HeaderBar::builder()
            .show_end_title_buttons(true)
            .title_widget(&Label::new(Some("Scriptinator")))
            .build();

        let start_stop_button = Button::builder()
            .label("Start Recording")
            .build();
        header_bar.pack_start(&start_stop_button);

        let menu_button = MenuButton::builder()
            .icon_name("open-menu-symbolic")
            .build();

        let settings_box = GtkBox::new(Orientation::Vertical, 12);
        settings_box.set_margin_start(12);
        settings_box.set_margin_end(12);
        settings_box.set_margin_top(12);
        settings_box.set_margin_bottom(12);

        let audio_group = PreferencesGroup::new();
        audio_group.set_title("Audio Settings");

        let audio_row = ActionRow::new();
        audio_row.set_title("Audio Input Device");

        let device_list = StringList::new(&[]);
        for device in &devices {
            device_list.append(device);
        }
        let device_dropdown = DropDown::builder()
            .model(&device_list)
            .selected(0)
            .valign(Align::Center)
            .build();

        audio_row.add_suffix(&device_dropdown);
        audio_group.add(&audio_row);

        let app_row = ActionRow::new();
        app_row.set_title("Running Applications");
        app_row.set_subtitle("Select an application to record its audio");

        let app_list = StringList::new(&[]);
        app_list.append("None - Use device above");
        for (_id, name) in &applications {
            app_list.append(name);
        }
        let app_dropdown = DropDown::builder()
            .model(&app_list)
            .selected(0)
            .valign(Align::Center)
            .build();

        app_row.add_suffix(&app_dropdown);
        audio_group.add(&app_row);

        settings_box.append(&audio_group);

        let api_group = PreferencesGroup::new();
        api_group.set_title("KoboldCPP Settings");

        let url_row = ActionRow::new();
        url_row.set_title("KoboldCPP URL");
        let url_entry = Entry::new();
        url_entry.set_text(&settings.kobold_url.lock().unwrap());
        url_entry.set_valign(Align::Center);
        url_entry.set_width_chars(30);
        url_row.add_suffix(&url_entry);
        api_group.add(&url_row);

        settings_box.append(&api_group);

        let settings_popover = Popover::builder()
            .child(&settings_box)
            .build();
        menu_button.set_popover(Some(&settings_popover));
        header_bar.pack_end(&menu_button);

        main_vbox.append(&header_bar);

        let scrolled_window = ScrolledWindow::new();
        scrolled_window.set_vexpand(true);
        scrolled_window.set_margin_start(12);
        scrolled_window.set_margin_end(12);
        scrolled_window.set_margin_top(12);
        scrolled_window.set_margin_bottom(12);

        let cards_container = GtkBox::new(Orientation::Vertical, 8);
        scrolled_window.set_child(Some(&cards_container));
        main_vbox.append(&scrolled_window);

        let window = AdwApplicationWindow::builder()
            .application(app)
            .default_width(600)
            .default_height(700)
            .content(&main_vbox)
            .title("Scriptinator")
            .build();

        let provider = CssProvider::new();
        provider.load_from_data(
            r#"
            .card {
                padding: 16px;
                margin: 4px 0;
                border: 1px solid alpha(@borders, 0.3);
                border-radius: 8px;
                background-color: alpha(@theme_base_color, 0.1);
                backdrop-filter: blur(10px);
                box-shadow: 0 2px 8px alpha(@shadow_color, 0.16);
                transition: all 200ms ease;
            }
            .card:hover {
                background-color: alpha(@theme_base_color, 0.15);
                box-shadow: 0 4px 16px alpha(@shadow_color, 0.24);
            }
            .timestamp {
                font-size: 0.85em;
                color: alpha(@theme_fg_color, 0.6);
                font-weight: 500;
            }
            .translated-text {
                color: @theme_fg_color;
                font-weight: 500;
            }
            "#,
        );
        gtk::style_context_add_provider_for_display(
            &gdk::Display::default().expect("Could not connect to a display."),
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );

        let running_clone_for_close = running.clone();
        window.connect_close_request(move |_| {
            println!("Close request received, signaling worker thread to exit.");
            running_clone_for_close.store(false, Ordering::Relaxed);
            gtk::glib::Propagation::Proceed
        });

        window.present();

        let (tx, rx) = mpsc::channel::<(String, String)>();
        let rx = Arc::new(Mutex::new(rx));

        let rx_clone = Arc::clone(&rx);
        glib::timeout_add_local(Duration::from_millis(250), move || {
            let mut has_message = false;
            if let Ok(rx) = rx_clone.try_lock() {
                if let Ok((translated, timestamp)) = rx.try_recv() {
                    has_message = true;
                    let card = GtkBox::new(Orientation::Vertical, 8);
                    card.add_css_class("card");

                    let translated_label = Label::builder()
                        .label(&translated)
                        .wrap(true)
                        .xalign(0.0)
                        .css_classes(vec!["translated-text"])
                        .build();
                    card.append(&translated_label);

                    let timestamp_label = Label::builder()
                        .label(&timestamp)
                        .halign(Align::End)
                        .css_classes(vec!["timestamp"])
                        .build();
                    card.append(&timestamp_label);

                    cards_container.prepend(&card);
                }
            }
            glib::ControlFlow::Continue
        });

        let selected_device_index = Arc::new(AtomicUsize::new(0));
        let selected_app_index = Arc::new(AtomicUsize::new(0));

        let selected_device_index_clone = selected_device_index.clone();
        device_dropdown.connect_selected_notify(move |dropdown| {
            selected_device_index_clone.store(dropdown.selected() as usize, Ordering::Relaxed);
        });

        let selected_app_index_clone = selected_app_index.clone();
        app_dropdown.connect_selected_notify(move |dropdown| {
            selected_app_index_clone.store(dropdown.selected() as usize, Ordering::Relaxed);
        });

        let settings_clone_for_url_entry = Arc::clone(&settings);
        url_entry.connect_changed(move |entry| {
            *settings_clone_for_url_entry.kobold_url.lock().unwrap() = entry.text().to_string();
        });

        let (control_tx, control_rx) = mpsc::channel::<bool>();

        let running_clone_for_button = Arc::clone(&running);
        let control_tx_clone = control_tx.clone();
        start_stop_button.connect_clicked(move |button| {
            let is_running = running_clone_for_button.load(Ordering::Relaxed);
            if is_running {
                button.set_label("Start Recording");
                running_clone_for_button.store(false, Ordering::Relaxed);
                let _ = control_tx_clone.send(false);
            } else {
                button.set_label("Stop Recording");
                running_clone_for_button.store(true, Ordering::Relaxed);
                let _ = control_tx_clone.send(true);
            }
        });

        let running_clone_for_thread = Arc::clone(&running);
        let settings_clone_for_thread = Arc::clone(&settings);
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let mut audio_capture = AudioCapture::new();
            let default_device = "default".to_string();

            loop {
                match control_rx.recv() {
                    Ok(should_run) => {
                        if should_run {
                            let device_index = selected_device_index.load(Ordering::Relaxed);
                            let device = devices.get(device_index).unwrap_or(&default_device);

                            if let Err(e) = audio_capture.start_recording(device) {
                                println!("Failed to start recording: {}", e);
                                continue;
                            }

                            while running_clone_for_thread.load(Ordering::Relaxed) {
                                if let Some(audio_data) = audio_capture.read_audio_data() {
                                    let settings = Arc::clone(&settings_clone_for_thread);
                                    let tx = tx.clone();

                                    rt.spawn(async move {
                                        if let Some(translated) = translate_with_kobold(audio_data, &settings).await {
                                            if !translated.trim().is_empty() {
                                                let timestamp = Local::now().format("%H:%M:%S").to_string();
                                                let _ = tx.send((translated, timestamp));
                                            }
                                        }
                                    });
                                }

                                if let Ok(false) = control_rx.try_recv() {
                                    break;
                                }
                            }

                            audio_capture.stop_recording();
                        }
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
        });
    });

    app.run();
}
