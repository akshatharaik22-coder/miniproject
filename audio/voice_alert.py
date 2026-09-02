import pyttsx3

# Initialize engine
engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)


def speak_objects(objects):

    if len(objects) == 0:
        text = "No obstacle detected."
    else:
        text = "Warning. "

        for obj in objects:
            text += obj + " detected. "

    print(text)

    engine.say(text)
    engine.runAndWait()


# -------------------------
# Testing
# -------------------------

if __name__ == "__main__":

    detected_objects = [
        "Person",
        "Chair",
        "Dog"
    ]

    speak_objects(detected_objects)