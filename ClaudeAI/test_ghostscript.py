import subprocess

GHOSTSCRIPT_COMMAND = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"

try:
    result = subprocess.run(
        [GHOSTSCRIPT_COMMAND, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("Ghostscript gefunden:", result.stdout.decode().strip())
except Exception as e:
    print("Fehler beim Ausführen von Ghostscript:")
    print(e)


try:
    subprocess.run(
        [r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe", "--version"],
        check=True
    )
    print("Ghostscript läuft!")
except Exception as e:
    print(f"Fehler: {e}")
