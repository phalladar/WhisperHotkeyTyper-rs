# WhisperHotkeyTyper-rs

**WhisperHotkeyTyper-rs** is a Rust desktop application that provides "push-to-talk" voice transcription using OpenAI's Whisper model (via `whisper.cpp`). Press and hold a hotkey (Left Ctrl + Left Alt + S by default) to record your voice, and upon release, the app transcribes the audio and types the result into your currently active window.

It's designed for quick, local voice dictation without relying on cloud services.

## Features
*   **Push-to-Talk Transcription:** Press and hold **Left Ctrl + Left Alt + S** to record, release to transcribe.

*   **Automatic Typing:** Transcribed text is automatically typed into the active application using the `enigo` crate.
*   **Local Processing:** Uses `whisper.cpp` (via the `whisper-rs` bindings) for on-device transcription.
*   **GPU Acceleration:** Leverages GPU for faster transcription if `whisper.cpp` is compiled with CUDA/Metal support and a compatible GPU is available.
*   **Simple GUI:** Displays status (Idle, Recording, Transcribing, Error) and the latest transcription using `eframe`/`egui`.
*   **Audio Input & Resampling:** Captures audio using `cpal` (configured for 48kHz mono) and resamples to 16kHz for Whisper using `rubato`.
*   **Cross-Platform Core:** Built with Rust, aiming for cross-platform compatibility (primarily tested on Windows). Hotkey listening uses `device_query`.
*   **Logging:** Detailed logs are saved to `app.log` using `log4rs`.

## Prerequisites

1.  **Rust Toolchain:** Install Rust from [rustup.rs](https://rustup.rs/).
2.  **Whisper Model:** You need a `whisper.cpp` compatible model in GGML format.
    *   Download models from the official `whisper.cpp` Hugging Face repository:
        **[https://huggingface.co/ggerganov/whisper.cpp/tree/main](https://huggingface.co/ggerganov/whisper.cpp/tree/main)**
    *   This application is currently configured to use `ggml-medium.en-q8_0.bin` by default. Other models should work, but you might need to adjust the `model_path_str` in `src/main.rs`.
3.  **Audio Input Device:** A working microphone.
4.  **(Optional) For GPU Acceleration:**
    *   **NVIDIA GPU:** CUDA Toolkit installed.
    *   **AMD GPU (Linux):** ROCm stack installed.
    *   **Apple Silicon:** Metal support is usually available out-of-the-box.
    *   The `whisper-rs` crate (and its underlying `whisper.cpp` dependency) must be compiled with the appropriate GPU feature flags (e.g., `cuda`, `metal`). This might happen automatically if the build environment is set up correctly. Consult `whisper-rs` documentation for specifics.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/phalladar/WhisperHotkeyTyper-rs.git
    cd WhisperHotkeyTyper-rs
    ```

2.  **Download the Whisper Model:**
    *   Create a `models` directory in the project root:
        ```bash
        mkdir models
        ```
    *   Download your chosen GGML model (e.g., `ggml-medium.en-q8_0.bin`) from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main) and place it inside the `models` directory.
        The path should look like: `./models/ggml-medium.en-q8_0.bin`

3.  **Build the application:**
    ```bash
    cargo build --release
    ```
    For GPU support with CUDA, you might need to ensure `whisper-rs` picks up your CUDA installation (e.g., via `CUDA_PATH` environment variable). Check the `whisper-rs` documentation for specific build instructions if you encounter issues with GPU acceleration.

## Usage

1.  Run the executable from the `target/release` directory:
    ```bash
    # On Linux/macOS
    ./target/release/WhisperHotkeyTyper-rs
    # On Windows
    .\target\release\WhisperHotkeyTyper-rs.exe
    ```
2.  The application window will appear, showing the status "Idle. Press and hold Left Ctrl + Left Alt + S to record."
3.  Click into the application where you want to type (e.g., a text editor, browser).
4.  **Press and hold the `Left Ctrl + Left Alt + S` keys.** The status will change to "Recording...".
5.  Speak clearly into your microphone.
6.  **Release the `Left Ctrl + Left Alt + S` keys.** The status will change to "Processing..." and then "Transcribing...".
7.  Once transcription is complete, the text will be automatically typed into your active window. The GUI will also display the transcribed text.
8.  The status will update to "Transcription complete. Press and hold Left Ctrl + Left Alt + S to record again."

## Configuration (In Code)

Currently, some settings are hardcoded in `src/main.rs`:

*   **Model Path:**
    ```rust
    let model_path_str = "./models/ggml-medium.en-q8_0.bin";
    ```
*   **Transcription Language:**
    ```rust
    params.set_language(Some("en")); // Set to your desired language code
    ```
*   **Hotkey:** The Left Ctrl + Left Alt + S keys are used via `device_query`. Modifying this requires changing the `Keycode::LControl`, `Keycode::LAlt`, and `Keycode::S` checks in the `hotkey_thread`.

## Troubleshooting

*   **"Model file not found" error:**
    *   Ensure the model (e.g., `ggml-medium.en-q8_0.bin`) is correctly placed in a `models` subfolder relative to where you *run* the executable. If you run from the project root (`cargo run --release`), it's `./models/`. If you run directly from `target/release/`, you might need to copy the `models` folder there or adjust the path in the code.
    *   Check the `app.log` file for the exact path being checked.
*   **"No input audio device" or audio errors:**
    *   Ensure your microphone is connected and selected as the default input device in your system settings.
    *   Check `app.log` for more specific `cpal` errors.
*   **Text not typing / typing in the wrong window:**
    *   Make sure the desired target window is active and focused *before* you release the Left Ctrl + Left Alt + S keys.
    *   Some applications or system UI elements might not accept simulated input from `enigo` correctly.
*   **Poor transcription quality:**
    *   Try a larger/better quality Whisper model.
    *   Ensure your microphone input is clear and not too noisy.
    *   Check `app.log` for any errors during audio processing or transcription.
*   **GPU not being used:**
    *   Verify your GPU drivers and (if applicable) CUDA Toolkit are correctly installed and up to date.
    *   Check the console output when the application starts (or `app.log`). `whisper.cpp` usually prints messages about GPU detection (e.g., "found CUDA device", "using Metal").
    *   You may need to recompile `whisper-rs` (which rebuilds `whisper.cpp`) with specific features if auto-detection fails. Refer to the `whisper-rs` crate documentation.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, features, or improvements.

Some ideas for future enhancements:
*   Configurable hotkey.
*   Model selection from the GUI.
*   Language selection from the GUI.
*   System tray icon and background operation.
*   More robust error handling and user feedback.

## License

This project is licensed under the MIT License. (You should add a `LICENSE` file with the MIT license text).

## Acknowledgements

*   [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model.
*   [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) for the C++ port.
*   [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) for the Rust bindings.
*   The Rust community and the developers of crates like `eframe`, `cpal`, `rubato`, `device_query`, and `enigo`.
