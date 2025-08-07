#![windows_subsystem = "windows"]
use eframe::egui;
use std::sync::{Arc, Mutex};
use crossbeam_channel::{unbounded, Sender, Receiver};
use std::thread;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(target_os = "macos")]
use global_hotkey::{GlobalHotKeyManager, hotkey::{HotKey, Code, Modifiers}};

#[cfg(not(target_os = "macos"))]
use device_query::{DeviceState, Keycode};

use cpal::traits::{HostTrait, DeviceTrait, StreamTrait};
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy, WhisperContextParameters};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction}; // For resampling
use enigo::{Enigo, Settings, Keyboard}; // Added for typing output
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;

// --- Messages for thread communication ---
#[derive(Debug)]
enum AppMessage {
    TranscriptionResult(String),
    StatusUpdate(String),
    AudioData(Vec<f32>),
}

// --- Application State ---
struct WhisperApp {
    status: String,
    transcribed_text: String,
    is_recording: bool, 
    model_path: PathBuf,
    rx_from_others: Receiver<AppMessage>,
    _audio_control_tx: Arc<Mutex<Option<Sender<AudioControlMessage>>>>,
}

#[derive(Debug)]
enum AudioControlMessage {
    StartRecording,
    StopAndProcess,
}


impl WhisperApp {
    fn new(cc: &eframe::CreationContext<'_>, model_path: PathBuf, rx_from_others: Receiver<AppMessage>, audio_control_tx: Arc<Mutex<Option<Sender<AudioControlMessage>>>>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            status: "Idle. Press and hold Ctrl + Alt + S to record.".to_string(),
            transcribed_text: "".to_string(),
            is_recording: false,
            model_path,
            rx_from_others,
            _audio_control_tx: audio_control_tx,
        }
    }

    fn handle_messages(&mut self) {
        while let Ok(msg) = self.rx_from_others.try_recv() {
            log::debug!("GUI received message: {:?}", msg);
            match msg {
                AppMessage::TranscriptionResult(text) => {
                    self.transcribed_text = text;
                    self.status = "Transcription complete. Press and hold Ctrl + Alt + S to record again.".to_string();
                    self.is_recording = false; 
                }
                AppMessage::StatusUpdate(status) => {
                    if status.contains("Recording...") {
                        self.is_recording = true;
                         self.transcribed_text = "".to_string(); 
                    } else if status.contains("Processing...") || status.contains("Idle") || status.contains("Error") || status.contains("complete") || status.contains("Ready") {
                        self.is_recording = false;
                    }
                    self.status = status; 
                }
                AppMessage::AudioData(_) => {
                    log::warn!("GUI received unexpected AudioData message");
                }
            }
        }
    }
}

impl eframe::App for WhisperApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_messages();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Whisper Rust CUDA App");
            ui.separator();
            ui.label(format!("Model: {}", self.model_path.file_name().unwrap_or_default().to_string_lossy()));

            ui.horizontal(|ui| {
                ui.label(format!("Status: {}", self.status));
                if self.is_recording || self.status.contains("Transcribing...") || self.status.contains("Processing...") {
                    ui.add(egui::Spinner::new());
                }
            });

            ui.separator();
            ui.label("Transcription:");
            egui::ScrollArea::vertical().auto_shrink([false; 2]).stick_to_bottom(true).show(ui, |ui| {
                ui.label(self.transcribed_text.as_str());
            });
        });
        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Configure File Logging ---
    // Determine log path relative to the executable
    let mut exe_log_path = std::env::current_exe()?;
    exe_log_path.pop(); // Remove the executable name, leaving the directory
    let log_path = exe_log_path.join("app.log");

    // Log pattern: Date/Time [Level] [Target] Message
    let encoder = PatternEncoder::new("{d(%Y-%m-%d %H:%M:%S%.3f)} [{l}] [{t}] - {m}{n}");
    let file_appender = FileAppender::builder()
        .encoder(Box::new(encoder))
        .build(log_path)
        .expect("Failed to build file appender.");
    let config = Config::builder()
        .appender(Appender::builder().build("file", Box::new(file_appender)))
        .build(Root::builder().appender("file").build(LevelFilter::Debug)) // Set default log level to Debug for troubleshooting
        .expect("Failed to build log config.");
    log4rs::init_config(config).expect("Failed to initialize logger.");

    log::info!("--- Application Starting ---"); // Add a start marker

    // Determine model path relative to the executable
    let mut exe_path = std::env::current_exe()?;
    exe_path.pop(); // Remove the executable name, leaving the directory
    let model_path = exe_path.join("models/ggml-medium.en-q8_0.bin");

    // log::info!("Looking for model at: {}", model_path.display()); // For debugging

    if !model_path.exists() {
        let current_dir = std::env::current_dir()?;
        log::error!("Model file not found at: '{}'", model_path.display());
        log::error!("Execution directory: '{}'", current_dir.display());
        log::error!("Please ensure the model is in a 'models' subfolder relative to the execution directory.");
        log::error!("Attempted full path: {}", std::fs::canonicalize(&model_path).unwrap_or_else(|_| model_path.clone()).display());
        return Err(format!("Model file not found: {}", model_path.display()).into());
    }
    log::info!("Using model: {}", model_path.display());


    let (tx_main_thread, rx_main_thread) = unbounded::<AppMessage>();
    let (tx_audio_control, rx_audio_control) = unbounded::<AudioControlMessage>();
    let (tx_audio_data_to_whisper, rx_audio_data_from_audio) = unbounded::<AppMessage>();

    let shared_audio_control_tx = Arc::new(Mutex::new(Some(tx_audio_control)));

    // --- Spawn Whisper Processing Thread ---
    let tx_main_whisper_clone = tx_main_thread.clone();
    let whisper_model_path_clone = model_path.clone();
    thread::Builder::new().name("whisper_thread".into()).spawn(move || {
        log::info!("Whisper thread started. Attempting to load model: {}", whisper_model_path_clone.display());

        let context_params = WhisperContextParameters::default();
        log::info!("Attempting to load model with GPU support (use_gpu: {} by default from params)", context_params.use_gpu);

        let context = match WhisperContext::new_with_params(&whisper_model_path_clone.to_string_lossy(), context_params) {
            Ok(ctx) => {
                log::info!("Whisper model loaded. Check console/stderr for whisper.cpp messages about GPU usage.");
                ctx
            },
            Err(e) => {
                log::error!("Failed to load whisper model: {}. Path: {}", e, whisper_model_path_clone.display());
                let _ = tx_main_whisper_clone.send(AppMessage::StatusUpdate(format!("Error: Model load failed: {}", e)));
                return;
            }
        };

        while let Ok(AppMessage::AudioData(audio_data_f32)) = rx_audio_data_from_audio.recv() {
            log::info!("Whisper thread received audio data ({} samples). Processing...", audio_data_f32.len());
            if tx_main_whisper_clone.send(AppMessage::StatusUpdate("Transcribing...".to_string())).is_err() {
                log::error!("Whisper thread: Failed to send status to main. Channel closed?");
                return;
            }

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
            params.set_language(Some("en"));
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            let mut state = context.create_state().expect("Failed to create whisper state");
            match state.full(params, &audio_data_f32[..]) {
                Ok(_) => {
                    let num_segments = state.full_n_segments().expect("Failed to get segment count");
                    log::info!("Transcription reported success. Found {} segments.", num_segments);
                    let mut result_text = String::new();
                    for i in 0..num_segments {
                        match state.full_get_segment_text(i) {
                            Ok(segment) => {
                                log::info!("Segment {}: '{}'", i, segment);
                                result_text.push_str(&segment);
                            }
                            Err(e) => {
                                log::error!("Failed to get segment {} text: {}", i, e);
                            }
                        }
                    }
                    log::info!("Combined result before trim: '{}'", result_text);
                    let final_text = result_text.trim().to_string();
                    // Clone the text *before* sending it to the main thread
                    let text_for_main_thread = final_text.clone(); 
                    if tx_main_whisper_clone.send(AppMessage::TranscriptionResult(text_for_main_thread)).is_err() {
                         log::error!("Whisper thread: Failed to send result to main. Channel closed?");
                         return;
                    }
                    // --- Type the transcription result ---
                    if !final_text.is_empty() {
                        log::info!("Whisper thread: Attempting to type out transcription...");
                        match Enigo::new(&Settings::default()) {
                            Ok(mut enigo) => {
                                // Type character by character with a small delay for terminal compatibility
                                for ch in final_text.chars() {
                                    match enigo.text(&ch.to_string()) {
                                        Ok(_) => {
                                            // Small delay between characters (1ms) for terminal compatibility
                                            std::thread::sleep(Duration::from_millis(1));
                                        },
                                        Err(e) => {
                                            log::error!("Whisper thread: Failed to type character '{}': {}", ch, e);
                                            break;
                                        }
                                    }
                                }
                                log::info!("Whisper thread: Successfully typed transcription.");
                            }
                            Err(e) => {
                                log::error!("Whisper thread: Failed to initialize enigo: {}", e);
                                // Optionally send an error status back to the main thread
                                let _ = tx_main_whisper_clone.send(AppMessage::StatusUpdate(format!("Typing Error: {}", e)));
                            }
                        }
                    } else {
                        log::info!("Whisper thread: Transcription result was empty, not typing anything.");
                    }
                }
                Err(e) => {
                    log::error!("Whisper transcription failed: {}", e);
                    let _ = tx_main_whisper_clone.send(AppMessage::StatusUpdate(format!("Transcription Error: {}", e)));
                }
            }
        }
        log::info!("Whisper thread finished.");
    })?;


    // --- Spawn Audio Recording Thread ---
    let tx_main_audio_clone = tx_main_thread.clone();
    let audio_capture_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let rx_audio_control_clone = rx_audio_control;
    let tx_audio_data_to_whisper_clone = tx_audio_data_to_whisper; 
    thread::Builder::new().name("audio_thread".into()).spawn(move || {
        // Use the cloned channels inside the audio thread
        let rx_audio_control = rx_audio_control_clone;
        let tx_audio_data_to_whisper = tx_audio_data_to_whisper_clone;
        log::info!("Audio thread started.");
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                log::error!("No input audio device available.");
                let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Error: No input device.".to_string()));
                return;
            }
        };
        log::info!("Using default input device: {}", device.name().unwrap_or_else(|_| "Unknown".into()));

        let supported_configs_iter = match device.supported_input_configs() {
            Ok(configs) => configs,
            Err(e) => {
                log::error!("Error querying supported input configs: {}", e);
                let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Error: Cannot query audio configs.".to_string()));
                return;
            }
        };

        log::info!("Supported input configurations:");
        let mut available_configs = Vec::new();
        for cfg_range in supported_configs_iter {
            log::info!("- Channels: {}, Sample Rate: {:?}-{:?}, Format: {:?}", 
                cfg_range.channels(), 
                cfg_range.min_sample_rate(), 
                cfg_range.max_sample_rate(), 
                cfg_range.sample_format()
            );
            available_configs.push(cfg_range);
        }
        
        let desired_sr = cpal::SampleRate(48000);
        let chosen_config_range = available_configs.into_iter()
            .find(|config| {
                config.channels() == 1 && // <-- Change to Mono
                config.min_sample_rate() <= desired_sr &&
                desired_sr <= config.max_sample_rate() &&
                config.sample_format() == cpal::SampleFormat::F32 // Target F32
            });

        let config = match chosen_config_range {
            Some(config_range) => {
                log::info!("Using supported 48kHz Mono F32 config: {:?}", config_range);
                config_range.with_sample_rate(desired_sr).config()
            }
            None => {
                log::error!("No supported 48kHz Mono F32 config found. Please check logs.");
                let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Error: No suitable F32 audio config.".to_string()));
                return; 
            }
        };

        log::info!("Selected config for stream: Channels={}, Rate={}, Buffer={:?}", 
                 config.channels, config.sample_rate.0, config.buffer_size);

        let mut audio_stream_opt: Option<cpal::Stream> = None;
        let err_fn_arc = Arc::new(tx_main_audio_clone.clone());

        loop {
            match rx_audio_control.recv() {
                Ok(AudioControlMessage::StartRecording) => {
                    if let Some(stream) = audio_stream_opt.take() {
                        log::warn!("StartRecording received while a stream was already active. Stopping previous one.");
                        if let Err(e) = stream.pause() { log::warn!("Failed to pause previous stream: {}", e); }
                        drop(stream);
                    }
                    log::info!("Audio thread: StartRecording signal received.");
                    { 
                        let mut buffer_lock = audio_capture_buffer.lock().unwrap();
                        buffer_lock.clear();
                    }

                    let buffer_for_callback = Arc::clone(&audio_capture_buffer);
                    let err_fn_capture = Arc::clone(&err_fn_arc); 

                    let input_stream = match device.build_input_stream(
                        &config, 
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            if data.is_empty() {
                                log::debug!("Audio callback: Received empty buffer");
                            } else {
                                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                log::debug!("Audio callback: Received {} f32 samples, min={}, max={}", data.len(), min_val, max_val);
                            }
                            // --- Store f32 data directly ---
                            if let Ok(mut buffer_lock) = buffer_for_callback.lock() {
                                buffer_lock.extend_from_slice(data); // Store f32 data directly
                            } else { log::error!("Audio callback: Failed to lock audio buffer for writing."); }
                        },
                        move |err| {
                            log::error!("Audio stream error: {}", err);
                            let _ = err_fn_capture.send(AppMessage::StatusUpdate(format!("Audio Error: {}", err)));
                        },
                        None,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            log::error!("Failed to build audio input stream: {}", e);
                            let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate(format!("Audio Build Error: {}", e)));
                            continue;
                        }
                    };
                    if let Err(e) = input_stream.play() {
                        log::error!("Failed to play audio stream: {}", e);
                        let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate(format!("Audio Play Error: {}", e)));
                        continue;
                    }
                    audio_stream_opt = Some(input_stream);
                    log::info!("Audio recording started.");
                    let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Recording...".to_string()));
                }
                Ok(AudioControlMessage::StopAndProcess) => {
                    log::info!("Audio thread: StopAndProcess signal received.");
                    if let Some(stream) = audio_stream_opt.take() { 
                        if let Err(e) = stream.pause() { log::warn!("Failed to pause audio stream: {}", e); }
                        drop(stream);
                        log::info!("Audio recording stopped.");

                        let raw_stereo_audio_data = {
                            let mut buffer_lock = audio_capture_buffer.lock().unwrap();
                            let data_to_process = buffer_lock.clone();
                            buffer_lock.clear();
                            data_to_process
                        };

                        if !raw_stereo_audio_data.is_empty() {
                            log::info!("Processing {} raw stereo f32 samples.", raw_stereo_audio_data.len());

                            // We are now getting mono f32 data directly from the callback's buffer
                            let mono_48k_data = raw_stereo_audio_data; // Use the data directly
                            log::info!("Processing {} raw mono f32 samples.", mono_48k_data.len());

                            // 2. Resample from 48kHz mono to 16kHz mono
                            let input_sr = 48000.0;
                            let output_sr = 16000.0;
                            let sinc_len = 256;
                            let params = SincInterpolationParameters {
                                sinc_len,
                                f_cutoff: 0.95,
                                interpolation: SincInterpolationType::Linear,
                                oversampling_factor: 256, // High quality
                                window: WindowFunction::BlackmanHarris2, 
                            };
                            // Create resampler (fixed number of input channels, mono)
                            let mut resampler = SincFixedIn::<f32>::new(
                                output_sr / input_sr, 
                                2.0,                  
                                params,
                                mono_48k_data.len(),  
                                1, 
                            ).expect("Failed to create resampler");

                            // Prepare input for resampler (Vec<Vec<f32>>)
                            let waves_in = vec![mono_48k_data];
                            match resampler.process(&waves_in, None) {
                                Ok(resampled_waves) => {
                                    let final_16k_mono_data = resampled_waves.into_iter().next().unwrap_or_default();
                                    log::info!("Resampled to {} 16kHz mono samples. Sending to Whisper.", final_16k_mono_data.len());
                                    
                                    if !final_16k_mono_data.is_empty() {
                                        if tx_audio_data_to_whisper.send(AppMessage::AudioData(final_16k_mono_data)).is_err() {
                                            log::error!("Audio thread: Failed to send audio data to whisper. Channel closed?");
                                        }
                                    } else {
                                        log::warn!("Resampled audio is empty. Not sending to Whisper.");
                                        let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("No audio after resampling.".to_string()));
                                    }
                                }
                                Err(e) => {
                                    log::error!("Audio resampling failed: {:?}", e);
                                    let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Error: Resampling failed.".to_string()));
                                }
                            }
                        } else {
                            log::warn!("No audio data recorded to send.");
                            let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("No audio data recorded.".to_string()));
                        }
                    } else {
                        log::warn!("Stop signal received but no active stream found.");
                         let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Ready. (No stream was active)".to_string()));
                    }
                     let _ = tx_main_audio_clone.send(AppMessage::StatusUpdate("Processing... Complete. Ready for next Ctrl + Alt + S.".to_string())); // Updated status
                }
                Err(_) => {
                    log::info!("Audio control channel closed. Exiting audio thread.");
                    break;
                }
            }
        }
        log::info!("Audio thread finished.");
    })?;

    // --- Spawn Hotkey Listener Thread ---
    #[cfg(target_os = "macos")]
    {
        let tx_main_hotkey_clone = tx_main_thread.clone();
        let audio_control_tx_hotkey_clone = Arc::clone(&shared_audio_control_tx);
        
        thread::Builder::new().name("hotkey_thread".into()).spawn(move || {
            log::info!("macOS Hotkey thread started (using global-hotkey).");
            
            // Create the hotkey manager
            let manager = match GlobalHotKeyManager::new() {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Failed to create GlobalHotKeyManager: {}", e);
                    let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Hotkey Error: {}", e)));
                    return;
                }
            };
            
            // Register Control + Alt + S
            let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyS);
            
            if let Err(e) = manager.register(hotkey) {
                log::error!("Failed to register hotkey: {}", e);
                let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Failed to register hotkey: {}", e)));
                return;
            }
            
            log::info!("Successfully registered Control + Alt + S hotkey");
            
            let mut is_recording = false;
            
            loop {
                // Check for hotkey events
                if let Ok(event) = global_hotkey::GlobalHotKeyEvent::receiver().try_recv() {
                    log::info!("Hotkey event received: {:?}", event);
                    
                    if event.id == hotkey.id() {
                        if !is_recording {
                            // Start recording
                            is_recording = true;
                            log::info!("macOS: Control + Alt + S Pressed - Starting recording");
                            
                            if let Some(tx) = audio_control_tx_hotkey_clone.lock().unwrap().as_ref() {
                                if let Err(e) = tx.send(AudioControlMessage::StartRecording) {
                                    log::error!("Failed to send StartRecording: {}", e);
                                    let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Error: {}", e)));
                                } else {
                                    let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Recording...".to_string()));
                                }
                            }
                        } else {
                            // Stop recording
                            is_recording = false;
                            log::info!("macOS: Control + Alt + S Pressed - Stopping recording");
                            
                            if let Some(tx) = audio_control_tx_hotkey_clone.lock().unwrap().as_ref() {
                                if let Err(e) = tx.send(AudioControlMessage::StopAndProcess) {
                                    log::error!("Failed to send StopAndProcess: {}", e);
                                    let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Error: {}", e)));
                                } else {
                                    let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Processing...".to_string()));
                                }
                            }
                        }
                    }
                }
                
                std::thread::sleep(Duration::from_millis(10));
            }
        })?;
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        let tx_main_hotkey_clone = tx_main_thread.clone();
        let audio_control_tx_hotkey_clone = Arc::clone(&shared_audio_control_tx);
        
        thread::Builder::new().name("hotkey_thread".into()).spawn(move || {
            log::info!("DeviceQuery Hotkey thread started.");
            let device_state = DeviceState::new();
            let mut last_hotkey_pressed = false;

            loop {
                let keys = device_state.query_keymap();
                // Check for Control + Alt + S (works better cross-platform)
                let ctrl_pressed = keys.contains(&Keycode::LControl) || keys.contains(&Keycode::RControl);
                let alt_pressed = keys.contains(&Keycode::LAlt) || keys.contains(&Keycode::RAlt);
                let s_pressed = keys.contains(&Keycode::S);
                let hotkey_pressed = ctrl_pressed && alt_pressed && s_pressed;

                if hotkey_pressed && !last_hotkey_pressed {
                    log::info!("DeviceQuery: Control + Alt + S Pressed"); 
                    if let Some(tx) = audio_control_tx_hotkey_clone.lock().unwrap().as_ref() {
                         if let Err(e) = tx.send(AudioControlMessage::StartRecording) {
                             log::error!("DeviceQuery Hotkey thread: Failed to send StartRecording: {}", e);
                             let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Error: {}", e)));
                         } else {
                             let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Recording...".to_string()));
                         }
                     } else {
                          log::error!("DeviceQuery Hotkey thread: Audio control channel unavailable for StartRecording.");
                          let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Error: Audio control unavailable".to_string()));
                     }

                } else if !hotkey_pressed && last_hotkey_pressed {
                    log::info!("DeviceQuery: Control + Alt + S Released"); 
                     if let Some(tx) = audio_control_tx_hotkey_clone.lock().unwrap().as_ref() {
                         if let Err(e) = tx.send(AudioControlMessage::StopAndProcess) {
                             log::error!("DeviceQuery Hotkey thread: Failed to send StopAndProcess: {}", e);
                             let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate(format!("Error sending stop command: {}", e)));
                         } else {
                             let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Processing Hotkey Release...".to_string()));
                         }
                     } else {
                          log::error!("DeviceQuery Hotkey thread: Audio control channel unavailable for StopAndProcess.");
                          let _ = tx_main_hotkey_clone.send(AppMessage::StatusUpdate("Error: Audio control unavailable".to_string()));
                     }
                }

                last_hotkey_pressed = hotkey_pressed;

                thread::sleep(Duration::from_millis(50)); // Check ~20 times per second
            }
        })?;
    }

    log::info!("Starting GUI...");
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 400.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Whisper Rust CUDA App",
        native_options,
        Box::new(move |cc| {
            let app = WhisperApp::new(cc, model_path, rx_main_thread, shared_audio_control_tx);
            Box::new(app)
        }),
    )?;

    Ok(())
}
