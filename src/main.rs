use gtk::prelude::*;
use gtk::{Application, Box as GtkBox, Orientation, Label, ScrolledWindow, gdk, CssProvider, Align, DropDown, StringList, MenuButton, Popover, Button};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use gtk::glib;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use chrono::Local;
use std::io::{Write, Read, BufReader, BufRead};
use libadwaita::prelude::*;
use libadwaita::{HeaderBar, PreferencesGroup, ActionRow, ApplicationWindow as AdwApplicationWindow};
use std::collections::VecDeque;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use hf_hub::api::sync::Api;

// --- Runtime-configurable Settings ---
struct AppSettings {
    model_path: Mutex<Option<String>>,
    model_downloaded: AtomicBool,
}

impl AppSettings {
    fn new() -> Self {
        Self {
            model_path: Mutex::new(None),
            model_downloaded: AtomicBool::new(false),
        }
    }

    fn get_or_download_model(&self) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(path) = self.model_path.lock().unwrap().as_ref() {
            return Ok(path.clone());
        }

        println!("Downloading model from HuggingFace...");
        let api = Api::new()?;
        let repo = api.model("ggerganov/whisper.cpp".to_string());
        let model_file = repo.get("ggml-small-q8_0.bin")?;

        let path_str = model_file.to_string_lossy().to_string();
        *self.model_path.lock().unwrap() = Some(path_str.clone());
        self.model_downloaded.store(true, Ordering::Relaxed);

        println!("Model downloaded to: {}", path_str);
        Ok(path_str)
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

fn get_default_source() -> Option<String> {
    let output = Command::new("pactl")
        .args(["get-default-sink"])
        .output()
        .ok()?;

    let default_sink = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if default_sink.is_empty() {
        None
    } else {
        // For sinks, we want the monitor source
        Some(format!("{}.monitor", default_sink))
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

fn transcribe_with_whisper(audio_data: Vec<u8>, settings: &Arc<AppSettings>) -> Option<String> {
    // Convert raw PCM data to float samples
    let mut float_audio = Vec::new();
    for chunk in audio_data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
        float_audio.push(sample);
    }

    // Load model
    let model_path = match settings.get_or_download_model() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Failed to get model: {}", e);
            return None;
        }
    };

    let ctx = WhisperContext::new_with_params(
        &model_path,
        WhisperContextParameters::default()
    ).expect("failed to load model");

    // Create params with suppress_non_speech enabled
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    params.set_suppress_nst(false);
    params.set_suppress_blank(true);
    params.set_language(Some("en"));
    params.set_no_speech_thold(3.0);
    // Disable fallback temperatures (prevents hallucinations when uncertain)
    params.set_temperature_inc(0.0);
    // Set initial prompt to discourage common hallucinations
    params.set_initial_prompt("");
    // Set probability thresholds
    params.set_max_initial_ts(1.0);


    // Create state and run transcription
    let mut state = ctx.create_state().expect("failed to create state");
    state
        .full(params, &float_audio[..])
        .expect("failed to run model");

    // Collect results
    let mut result = String::new();
    for segment in state.as_iter() {
        result.push_str(&format!("{}", segment));
    }

    if result.trim().is_empty() {
        None
    } else {
        Some(result.trim().to_string())
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
        let default_source = get_default_source();
        let applications = get_running_applications();
        let main_vbox = GtkBox::new(Orientation::Vertical, 0);

        let title_label = Label::builder()
            .label("Scriptinator")
            .css_classes(vec!["heading"])
            .build();

        let header_bar = HeaderBar::builder()
            .show_end_title_buttons(true)
            .title_widget(&title_label)
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
        let mut default_index = 0;
        for (idx, device) in devices.iter().enumerate() {
            device_list.append(device);
            if let Some(ref default) = default_source {
                if device == default {
                    default_index = idx as u32;
                }
            }
        }
        let device_dropdown = DropDown::builder()
            .model(&device_list)
            .selected(default_index)
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
        api_group.set_title("Whisper Settings");

        let model_row = ActionRow::new();
        model_row.set_title("Whisper Model");
        model_row.set_subtitle("ggml-small-q8_0.bin from HuggingFace");

        let download_button = Button::builder()
            .label("Download Model")
            .valign(Align::Center)
            .build();

        let model_status_label = Label::builder()
            .label("Not downloaded")
            .valign(Align::Center)
            .build();

        // Channel for download status updates
        let (download_tx, download_rx) = mpsc::channel::<Result<String, String>>();

        let settings_clone = Arc::clone(&settings);
        download_button.connect_clicked(move |button| {
            button.set_sensitive(false);
            button.set_label("Downloading...");

            let settings = Arc::clone(&settings_clone);
            let tx = download_tx.clone();

            thread::spawn(move || {
                match settings.get_or_download_model() {
                    Ok(path) => {
                        let _ = tx.send(Ok(path));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e.to_string()));
                    }
                }
            });
        });

        // Handle download status updates on main thread
        let download_button_clone = download_button.clone();
        let status_label_clone = model_status_label.clone();
        glib::timeout_add_local(Duration::from_millis(100), move || {
            if let Ok(result) = download_rx.try_recv() {
                match result {
                    Ok(path) => {
                        download_button_clone.set_label("Downloaded ✓");
                        download_button_clone.set_sensitive(false);
                        status_label_clone.set_label(&format!("Ready: {}",
                            std::path::Path::new(&path)
                                .file_name()
                                .unwrap()
                                .to_string_lossy()));
                    }
                    Err(e) => {
                        download_button_clone.set_label("Download Model");
                        download_button_clone.set_sensitive(true);
                        status_label_clone.set_label(&format!("Error: {}", e));
                    }
                }
            }
            glib::ControlFlow::Continue
        });

        model_row.add_suffix(&model_status_label);
        model_row.add_suffix(&download_button);
        api_group.add(&model_row);

        settings_box.append(&api_group);

        let settings_popover = Popover::builder()
            .child(&settings_box)
            .build();
        menu_button.set_popover(Some(&settings_popover));
        header_bar.pack_end(&menu_button);

        main_vbox.append(&header_bar);

        // Create welcome screen
        let welcome_box = GtkBox::new(Orientation::Vertical, 12);
        welcome_box.set_valign(Align::Center);
        welcome_box.set_halign(Align::Center);
        welcome_box.set_margin_top(48);
        welcome_box.set_margin_bottom(48);
        welcome_box.set_margin_start(48);
        welcome_box.set_margin_end(48);

        let welcome_title = Label::builder()
            .label("Scriptinator")
            .css_classes(vec!["title-1"])
            .build();
        welcome_box.append(&welcome_title);

        let welcome_subtitle = Label::builder()
            .label("Real-time audio transcription with Whisper")
            .css_classes(vec!["title-2"])
            .build();
        welcome_box.append(&welcome_subtitle);

        let welcome_description = Label::builder()
            .label("To get started, download the Whisper model from settings (one-time setup) and click 'Start Recording' to begin transcription.")
            .wrap(true)
            .justify(gtk::Justification::Center)
            .css_classes(vec!["body"])
            .build();
        welcome_box.append(&welcome_description);

        // Create scrolled window for transcription results
        let scrolled_window = ScrolledWindow::new();
        scrolled_window.set_vexpand(true);
        scrolled_window.set_margin_start(12);
        scrolled_window.set_margin_end(12);
        scrolled_window.set_margin_top(12);
        scrolled_window.set_margin_bottom(12);

        let cards_container = GtkBox::new(Orientation::Vertical, 4);
        scrolled_window.set_child(Some(&cards_container));

        main_vbox.append(&welcome_box);
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
                    border: 1px solid alpha(#999, 0.3);
                    border-radius: 8px;
                    padding: 8px 10px;
                    margin-bottom: 0px;
                    margin-start: 8px;
                    background-color: alpha(#fff, 0.02);
                    display: block;
                    overflow: hidden;
                    box-sizing: border-box;
                }
                .timestamp {
                    color: alpha(#aaa, 0.8);
                    font-size: 0.8em;
                    margin-top: 4px;
                }
                .translated-text {
                    font-size: 11pt;
                    line-height: 1.4;
                    display: block;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                }
                .heading {
                    font-weight: bold;
                }
                .title-1 {
                    font-size: 24px;
                    font-weight: bold;
                }
                .title-2 {
                    font-size: 18px;
                    color: alpha(@theme_fg_color, 0.8);
                }
                .body {
                    font-size: 14px;
                    color: alpha(@theme_fg_color, 0.9);
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

        // Channel for sending transcriptions to UI
        let (tx, rx) = mpsc::channel::<(String, String)>();
        let rx = Arc::new(Mutex::new(rx));

        // Buffer to keep only 25 messages
        let message_buffer: Arc<Mutex<VecDeque<gtk::Widget>>> = Arc::new(Mutex::new(VecDeque::new()));

        let rx_clone = Arc::clone(&rx);
        let message_buffer_clone = Arc::clone(&message_buffer);
        glib::timeout_add_local(Duration::from_millis(250), move || {
            let mut has_message = false;
            if let Ok(rx) = rx_clone.try_lock() {
                if let Ok((translated, timestamp)) = rx.try_recv() {
                    has_message = true;

                    // Create new card
                    let card = GtkBox::new(Orientation::Vertical, 0);
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

                    // Add to buffer
                    if let Ok(mut buffer) = message_buffer_clone.try_lock() {
                        buffer.push_front(card.clone().into());

                        // Remove oldest if more than 25
                        if buffer.len() > 25 {
                            if let Some(old_card) = buffer.pop_back() {
                                cards_container.remove(&old_card);
                            }
                        }
                    }

                    // Add to UI
                    cards_container.prepend(&card);
                }
            }
            glib::ControlFlow::Continue
        });

        let selected_device_index = Arc::new(AtomicUsize::new(default_index as usize));
        let selected_app_index = Arc::new(AtomicUsize::new(0));

        let selected_device_index_clone = selected_device_index.clone();
        device_dropdown.connect_selected_notify(move |dropdown| {
            selected_device_index_clone.store(dropdown.selected() as usize, Ordering::Relaxed);
        });

        let selected_app_index_clone = selected_app_index.clone();
        app_dropdown.connect_selected_notify(move |dropdown| {
            selected_app_index_clone.store(dropdown.selected() as usize, Ordering::Relaxed);
        });

        let (control_tx, control_rx) = mpsc::channel::<bool>();

        let running_clone_for_button = Arc::clone(&running);
        let control_tx_clone = control_tx.clone();
        let welcome_box_clone = welcome_box.clone();
        let scrolled_window_clone = scrolled_window.clone();
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

                // Switch from welcome screen to transcription view
                main_vbox.remove(&welcome_box_clone);
                main_vbox.append(&scrolled_window_clone);
            }
        });

        let running_clone_for_thread = Arc::clone(&running);
        let settings_clone_for_thread = Arc::clone(&settings);
        thread::spawn(move || {
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

                                    // Process transcription in a separate thread to avoid blocking
                                    thread::spawn(move || {
                                        if let Some(translated) = transcribe_with_whisper(audio_data, &settings) {
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
