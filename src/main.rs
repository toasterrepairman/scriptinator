use gtk::prelude::*;
use gtk::{Application, Box, Orientation, Label, ScrolledWindow, gdk, CssProvider, Align, Entry, DropDown, StringList, MenuButton, Popover};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};
// Use the glib re-exported by gtk4 to avoid version conflicts
use gtk::glib;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use chrono::Local;
use reqwest;
use serde_json::json;
use tokio;
use std::io::Write;
use tempfile::NamedTempFile;
use reqwest::multipart;
// Import libadwaita prelude for extension traits
use libadwaita::prelude::*;
use libadwaita::{HeaderBar, PreferencesGroup, ActionRow, ApplicationWindow as AdwApplicationWindow};

// --- Runtime-configurable Settings ---
struct AppSettings {
    openai_url: Mutex<String>,
    transcription_model: Mutex<String>,
    chat_model: Mutex<String>,
}

impl AppSettings {
    fn new() -> Self {
        Self {
            openai_url: Mutex::new("http://localhost:11434".to_string()),
            transcription_model: Mutex::new("whisper".to_string()),
            chat_model: Mutex::new("llama3".to_string()),
        }
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

fn capture_audio_snippet(device: &str) -> Option<Vec<u8>> {
    println!("Capturing 8 seconds of audio from device: {}", device);
    let output = Command::new("ffmpeg")
        .args([
            "-f", "pulse", "-i", device, "-t", "8",
            "-f", "wav",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1",
        ])
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                if output.stdout.is_empty() {
                    println!("Warning: No audio data captured");
                    None
                } else {
                    Some(output.stdout)
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("FFmpeg error: {}", stderr);
                None
            }
        }
        Err(e) => {
            println!("Failed to run ffmpeg: {}", e);
            None
        }
    }
}

async fn transcribe_with_openai(audio_data: Vec<u8>, settings: &Arc<AppSettings>) -> Option<String> {
    let temp_file = NamedTempFile::new().ok()?;
    temp_file.as_file().write_all(&audio_data).ok()?;

    let file_part = multipart::Part::bytes(audio_data).file_name("audio.wav").mime_str("audio/wav").ok()?;

    let client = reqwest::Client::new();
    let url = format!("{}/v1/audio/transcriptions", settings.openai_url.lock().unwrap());
    let model = settings.transcription_model.lock().unwrap().clone();

    let form = multipart::Form::new()
        .text("model", model)
        .text("response_format", "json")
        .part("file", file_part);

    let response = client.post(&url).multipart(form).send().await.ok()?;
    if response.status().is_success() {
        let json: serde_json::Value = response.json().await.ok()?;
        json.get("text").and_then(|v| v.as_str()).map(|s| s.trim().to_string())
    } else {
        println!("Transcription HTTP error: {}", response.status());
        None
    }
}

async fn translate_text_with_openai(text: &str, settings: &Arc<AppSettings>) -> Option<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", settings.openai_url.lock().unwrap());
    let model = settings.chat_model.lock().unwrap().clone();

    let request_body = json!({
        "model": model,
        "messages": [{"role": "user", "content": format!("Translate any non-English portions of this text into English. Only provide the translated text as the response: {}", text)}]
    });

    let response = client.post(&url).json(&request_body).send().await.ok()?;
    if response.status().is_success() {
        let json: serde_json::Value = response.json().await.ok()?;
        json["choices"][0]["message"]["content"].as_str().map(|s| s.trim().to_string())
    } else {
        println!("Translation HTTP error: {}", response.status());
        None
    }
}

fn main() {
    let app = Application::builder()
        .application_id("com.toast.scriptinator")
        .build();

    app.connect_activate(|app| {
        let running = Arc::new(AtomicBool::new(true));
        let settings = Arc::new(AppSettings::new());
        let devices = get_audio_devices();
        let main_vbox = Box::new(Orientation::Vertical, 0);

        let header_bar = HeaderBar::builder()
            .show_end_title_buttons(true)
            .title_widget(&Label::new(Some("Scriptinator")))
            .build();

        // Create hamburger menu with settings
        let menu_button = MenuButton::builder()
            .icon_name("open-menu-symbolic")
            .build();

        // Create settings popover content
        let settings_box = Box::new(Orientation::Vertical, 12);
        settings_box.set_margin_start(12);
        settings_box.set_margin_end(12);
        settings_box.set_margin_top(12);
        settings_box.set_margin_bottom(12);

        // Audio Input Settings Group
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
        settings_box.append(&audio_group);

        // API Settings Group
        let api_group = PreferencesGroup::new();
        api_group.set_title("API Settings");

        let url_row = ActionRow::new();
        url_row.set_title("OpenAI-compatible URL");
        let url_entry = Entry::new();
        url_entry.set_text(&settings.openai_url.lock().unwrap());
        url_entry.set_valign(Align::Center);
        url_entry.set_width_chars(30);
        url_row.add_suffix(&url_entry);
        api_group.add(&url_row);

        let transcription_row = ActionRow::new();
        transcription_row.set_title("Transcription Model");
        let transcription_entry = Entry::new();
        transcription_entry.set_text(&settings.transcription_model.lock().unwrap());
        transcription_entry.set_valign(Align::Center);
        transcription_entry.set_width_chars(20);
        transcription_row.add_suffix(&transcription_entry);
        api_group.add(&transcription_row);

        let chat_row = ActionRow::new();
        chat_row.set_title("Chat Model");
        let chat_entry = Entry::new();
        chat_entry.set_text(&settings.chat_model.lock().unwrap());
        chat_entry.set_valign(Align::Center);
        chat_entry.set_width_chars(20);
        chat_row.add_suffix(&chat_entry);
        api_group.add(&chat_row);

        settings_box.append(&api_group);

        let settings_popover = Popover::builder()
            .child(&settings_box)
            .build();
        menu_button.set_popover(Some(&settings_popover));
        header_bar.pack_end(&menu_button);

        main_vbox.append(&header_bar);

        // Main content area - just the scrolled window with messages
        let scrolled_window = ScrolledWindow::new();
        scrolled_window.set_vexpand(true);
        scrolled_window.set_margin_start(12);
        scrolled_window.set_margin_end(12);
        scrolled_window.set_margin_top(12);
        scrolled_window.set_margin_bottom(12);

        let cards_container = Box::new(Orientation::Vertical, 8);
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
            .original-text {
                color: @theme_fg_color;
                font-weight: 500;
            }
            .translated-text {
                color: alpha(@theme_fg_color, 0.8);
                font-style: italic;
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

        let (tx, rx) = mpsc::channel::<(String, Option<String>, String)>();

        glib::idle_add_local(move || {
            if let Ok((original, translated, timestamp)) = rx.try_recv() {
                let card = Box::new(Orientation::Vertical, 8);
                card.add_css_class("card");

                let original_label = Label::builder()
                    .label(&original)
                    .wrap(true)
                    .xalign(0.0)
                    .css_classes(vec!["original-text"])
                    .build();
                card.append(&original_label);

                if let Some(translation) = translated {
                    let translated_label = Label::builder()
                        .label(&format!("→ {}", translation))
                        .wrap(true)
                        .xalign(0.0)
                        .css_classes(vec!["translated-text"])
                        .build();
                    card.append(&translated_label);
                }

                let timestamp_label = Label::builder()
                    .label(&timestamp)
                    .halign(Align::End)
                    .css_classes(vec!["timestamp"])
                    .build();
                card.append(&timestamp_label);

                cards_container.prepend(&card);
            }
            glib::ControlFlow::Continue
        });

        let selected_device_index = Arc::new(AtomicUsize::new(0));
        let selected_device_index_clone = selected_device_index.clone();
        device_dropdown.connect_selected_notify(move |dropdown| {
            selected_device_index_clone.store(dropdown.selected() as usize, Ordering::Relaxed);
        });

        // Settings change handlers
        let settings_clone_for_url_entry = Arc::clone(&settings);
        url_entry.connect_changed(move |entry| {
            *settings_clone_for_url_entry.openai_url.lock().unwrap() = entry.text().to_string();
        });

        let settings_clone_for_transcription = Arc::clone(&settings);
        transcription_entry.connect_changed(move |entry| {
            *settings_clone_for_transcription.transcription_model.lock().unwrap() = entry.text().to_string();
        });

        let settings_clone_for_chat = Arc::clone(&settings);
        chat_entry.connect_changed(move |entry| {
            *settings_clone_for_chat.chat_model.lock().unwrap() = entry.text().to_string();
        });

        let running_clone_for_thread = Arc::clone(&running);
        let settings_clone_for_thread = Arc::clone(&settings);
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let mut audio_buffer: Vec<u8> = Vec::new();
            let default_device = "default".to_string();

            while running_clone_for_thread.load(Ordering::Relaxed) {
                let device_index = selected_device_index.load(Ordering::Relaxed);
                let device = devices.get(device_index).unwrap_or(&default_device);

                if let Some(audio_snippet) = capture_audio_snippet(device) {
                    audio_buffer.extend_from_slice(&audio_snippet);
                    if audio_buffer.len() > 50_000 {
                        let audio_to_transcribe = audio_buffer.clone();
                        audio_buffer.clear();
                        let settings = Arc::clone(&settings_clone_for_thread);
                        let tx = tx.clone();
                        rt.spawn(async move {
                            if let Some(transcribed) = transcribe_with_openai(audio_to_transcribe, &settings).await {
                                if !transcribed.trim().is_empty() {
                                    let translated = translate_text_with_openai(&transcribed, &settings).await;
                                    let timestamp = Local::now().format("%H:%M:%S").to_string();
                                    let _ = tx.send((transcribed, translated, timestamp));
                                }
                            }
                        });
                    }
                }
                thread::sleep(Duration::from_millis(200));
            }
        });
    });

    app.run();
}
