import voicepad_core


def test_root_exports_current_audio_api() -> None:
    assert set(voicepad_core.__all__) == {
        "AudioPreProcessor",
        "AudioWindow",
        "DEFAULT_WAVEFORM_SPEC",
        "FileSource",
        "LiveWavRecording",
        "MicrophoneStream",
        "PreprocessedAudio",
        "RawAudio",
        "TARGET_SAMPLE_RATE",
        "WavArtifact",
        "WaveformSpec",
        "write_wav_atomic",
    }


def test_root_does_not_export_removed_transcription_api() -> None:
    assert not any(
        hasattr(voicepad_core, name)
        for name in (
            "Config",
            "StreamingTranscriber",
            "activate_model",
            "get_model",
            "transcribe",
            "transcribe_file",
        )
    )
