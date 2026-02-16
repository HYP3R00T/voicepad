from voicepad_core.diagnostics import gpu_diagnostics


def main() -> None:
    report = gpu_diagnostics()
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
