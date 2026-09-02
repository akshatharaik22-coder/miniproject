import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")


def speak_message(message):
    print("VOICE:", message)

    try:
        speaker.Speak(message)
    except Exception as e:
        print("Voice Error:", e)