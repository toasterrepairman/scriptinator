use gtk::prelude::*;
use gtk::{Application, ApplicationWindow, ComboBoxText, Box, Orientation, Label, ScrolledWindow, gdk, CssProvider};
use std::process::Command;
use std::thread;
use std::time::Duration;
use glib::{self, ControlFlow};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use chrono::Local;
use reqwest;
use serde_json::json;
use tokio;
use std::io::Write;
use tempfile::NamedTempFile;
use reqwest::multipart;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

// Configuration for the local OpenAI backend
const LOCAL_OPENAI_URL: &str = "http://localhost:11434"; // Default Ollama URL
const TRANSCRIPTION_MODEL: &str = "whisper"; // Model name for transcription
const CHAT_MODEL: &str = "llama3"; // Model name for chat completions/translation

fn get_audio_devices() -> Vec<String> {
    println!("Getting available audio devices...");

    let output = Command::new("pactl")
        .args(["list", "short", "sources"])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let devices: Vec<String> = stdout
                .lines()
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() >= 2 {
                        Some(parts[1].to_string())
                    } else {
                        None
                    }
                })
                .collect();

            println!("Found {} audio devices:", devices.len());
            for device in &devices {
                println!("  - {}", device);
            }

            devices
        }
        Err(e) => {
            println!("Error getting audio devices: {}", e);
            vec!["default".to_string()]
        }
    }
}

fn capture_audio_snippet(device: &str) -> Option<Vec<u8>> {
    println!("Capturing 5 seconds of audio from device: {}", device);

    let output = Command::new("ffmpeg")
        .args([
            "-f", "pulse", "-i", device, "-t", "5",
            "-f", "wav", // Changed to WAV for better compatibility
            "-ar", "16000", // 16kHz sample rate
            "-ac", "1", // Mono
            "pipe:1",
        ])
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                println!("Audio capture successful, {} bytes captured", output.stdout.len());
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

// Function to handle transcription, always requesting and parsing JSON
async fn transcribe_with_openai(audio_data: Vec<u8>) -> Option<String> {
    println!("Transcribing {} bytes of audio with local OpenAI backend...", audio_data.len());

    let mut temp_file = match NamedTempFile::new() {
        Ok(file) => file,
        Err(e) => {
            println!("Failed to create temporary file: {}", e);
            return None;
        }
    };

    if let Err(e) = temp_file.write_all(&audio_data) {
        println!("Failed to write audio data to temp file: {}", e);
        return None;
    }

    let file_part = match multipart::Part::bytes(audio_data).file_name("audio.wav").mime_str("audio/wav") {
        Ok(part) => part,
        Err(e) => {
            println!("Failed to create multipart part: {}", e);
            return None;
        }
    };

    let client = reqwest::Client::new();

    let form = multipart::Form::new()
        .text("model", TRANSCRIPTION_MODEL)
        .text("response_format", "json") // Explicitly request JSON format
        .part("file", file_part);

    let url = format!("{}/v1/audio/transcriptions", LOCAL_OPENAI_URL);

    match client
        .post(&url)
        .multipart(form)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                            let trimmed = text.trim();
                            println!("Transcription result: '{}'", trimmed);
                            if trimmed.is_empty() {
                                println!("Warning: Empty transcription result");
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        } else {
                            println!("No 'text' field in JSON response");
                            None
                        }
                    }
                    Err(e) => {
                        println!("Failed to parse JSON response: {}", e);
                        None
                    }
                }
            } else {
                println!("HTTP error: {}", response.status());
                None
            }
        }
        Err(e) => {
            println!("Request failed: {}", e);
            None
        }
    }
}

// New function to translate text using the chat completions endpoint
async fn translate_text_with_openai(text: &str) -> Option<String> {
    println!("Translating text with local OpenAI backend...");
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", LOCAL_OPENAI_URL);

    let request_body = json!({
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": format!("Translate any non-English portions of this text into English. Only provide the translated text as the response: {}", text)
            }
        ]
    });

    match client.post(&url).json(&request_body).send().await {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                            let trimmed = content.trim();
                            println!("Translation result: '{}'", trimmed);
                            if trimmed.is_empty() {
                                println!("Warning: Empty translation result");
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        } else {
                            println!("No content field in JSON response from chat model");
                            None
                        }
                    }
                    Err(e) => {
                        println!("Failed to parse JSON response from chat model: {}", e);
                        None
                    }
                }
            } else {
                println!("HTTP error from chat model: {}", response.status());
                println!("Response body: {}", response.text().await.unwrap_or_default());
                None
            }
        }
        Err(e) => {
            println!("Request failed for chat model: {}", e);
            None
        }
    }
}

fn main() {
    let app = Application::builder()
        .application_id("com.toast.scriptinator")
        .build();

    app.connect_activate(|app| {
        // --- Graceful Shutdown Flag ---
        let running = Arc::new(AtomicBool::new(true));

        // Get available audio devices
        let devices = get_audio_devices();

        // 1) Build UI
        let main_vbox = Box::new(Orientation::Vertical, 10);
        main_vbox.set_margin_top(10);
        main_vbox.set_margin_bottom(10);
        main_vbox.set_margin_start(10);
        main_vbox.set_margin_end(10);

        let scrolled_window = ScrolledWindow::new();
        scrolled_window.set_vexpand(true);
        let cards_container = Box::new(Orientation::Vertical, 0);
        scrolled_window.set_child(Some(&cards_container));
        main_vbox.append(&scrolled_window);

        let header_bar = gtk::HeaderBar::new();

        let header_device_combo = ComboBoxText::new();
        for device in &devices {
            header_device_combo.append_text(device);
        }
        header_device_combo.set_active(Some(0));
        header_bar.pack_end(&header_device_combo);

        let window = ApplicationWindow::builder()
            .application(app)
            .default_width(600)
            .default_height(400)
            .child(&main_vbox)
            .build();

        window.set_titlebar(Some(&header_bar));

        // --- CSS Styling ---
        let provider = CssProvider::new();
        provider.load_from_data("
            .card {
                padding: 12px;
                margin-bottom: 8px;
                border: 1px solid #d8dee9;
                border-radius: 4px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .timestamp {
                font-size: 0.8em;
                color: #888;
            }
        ");
        gtk::style_context_add_provider_for_display(
            &gdk::Display::default().expect("Could not connect to a display."),
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );

        // --- Graceful Shutdown Logic ---
        let running_clone_for_close = running.clone();
        window.connect_close_request(move |_| {
            println!("Close request received, signaling worker thread to exit.");
            running_clone_for_close.store(false, Ordering::Relaxed);
            gtk::glib::Propagation::Proceed
        });

        window.present();

        // 2) Channel - now sends (original_text, optional_translated_text, timestamp)
        let (tx, rx) = mpsc::channel::<(String, Option<String>, String)>();

        // Poll the receiver in the main thread to add new cards
        glib::idle_add_local(move || {
            if let Ok((original_text, translated_text, timestamp)) = rx.try_recv() {
                let card = Box::new(Orientation::Vertical, 5); // Changed to Vertical to stack labels
                card.add_css_class("card");

                let original_label = Label::new(Some(&original_text));
                original_label.set_wrap(true);
                original_label.set_xalign(0.0);
                original_label.set_halign(gtk::Align::Start);
                card.append(&original_label);

                if let Some(translation) = translated_text {
                    let translated_label = Label::new(Some(&format!("(Translated): {}", translation)));
                    translated_label.set_wrap(true);
                    translated_label.set_xalign(0.0);
                    translated_label.set_halign(gtk::Align::Start);
                    card.append(&translated_label);
                }

                let timestamp_label = Label::new(Some(&timestamp));
                timestamp_label.set_halign(gtk::Align::End);
                timestamp_label.set_hexpand(true);
                timestamp_label.add_css_class("timestamp");
                card.append(&timestamp_label);

                // Prepend the new card to add it to the top
                cards_container.prepend(&card);
            }
            ControlFlow::Continue
        });

        // 3) Worker thread with async runtime
        let selected_device_index = Arc::new(AtomicUsize::new(0));
        let selected_device_index_clone = selected_device_index.clone();
        let devices_clone = devices.clone();

        header_device_combo.connect_changed(move |combo| {
            if let Some(active) = combo.active() {
                selected_device_index_clone.store(active as usize, Ordering::Relaxed);
            }
        });

        let running_clone_for_thread = running.clone();
        std::thread::spawn(move || {
            // Create async runtime for the worker thread
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = tx.send((format!("Failed to create async runtime: {}", e), None, "".to_string()));
                    return;
                }
            };

            let tx_clone = tx.clone();
            println!("Testing connection to local OpenAI backend...");

            // Test the connection first
            rt.block_on(async {
                let client = reqwest::Client::new();
                let test_url = format!("{}/v1/models", LOCAL_OPENAI_URL);

                match client.get(&test_url).send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let _ = tx_clone.send(("Connected to local OpenAI backend. Starting audio capture...".to_string(), None, "".to_string()));
                        } else {
                            let _ = tx_clone.send((format!("Backend responded with status: {}", response.status()), None, "".to_string()));
                        }
                    }
                    Err(e) => {
                        let _ = tx_clone.send((format!("Failed to connect to backend at {}: {}", LOCAL_OPENAI_URL, e), None, "".to_string()));
                        return;
                    }
                }
            });

            let mut audio_buffer: Vec<u8> = Vec::new();

            // --- Main processing loop ---
            while running_clone_for_thread.load(Ordering::Relaxed) {
                let device_index = selected_device_index.load(Ordering::Relaxed);
                let default_device = "default".to_string();
                let device = devices_clone.get(device_index).unwrap_or(&default_device);

                if let Some(audio_snippet) = capture_audio_snippet(device) {
                    audio_buffer.extend_from_slice(&audio_snippet);

                    // Process when we have enough audio data (adjust threshold as needed)
                    if audio_buffer.len() > 200000 {
                        let audio_to_transcribe = audio_buffer.clone();
                        audio_buffer.clear();

                        // Run transcription and translation in async context
                        let (original_text, translated_text) = rt.block_on(async {
                            let transcription_result = transcribe_with_openai(audio_to_transcribe).await;
                            if let Some(transcribed_text) = &transcription_result {
                                let translation_result = translate_text_with_openai(transcribed_text).await;
                                (transcription_result, translation_result)
                            } else {
                                (transcription_result, None)
                            }
                        });

                        if let Some(text) = original_text {
                            if !text.trim().is_empty() {
                                let timestamp = Local::now().format("%H:%M:%S").to_string();
                                if tx.send((text, translated_text, timestamp)).is_err() {
                                    break; // Main thread has disconnected
                                }
                            }
                        }
                    }
                } else {
                    if tx.send(("Audio capture failed".to_string(), None, "".to_string())).is_err() {
                        break; // Main thread has disconnected
                    }
                }

                // Sleep for a short duration
                std::thread::sleep(Duration::from_millis(200));
            }
            println!("Worker thread has terminated.");
        });
    });

    app.run();
}
