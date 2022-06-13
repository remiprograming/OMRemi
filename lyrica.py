from midiutil import MIDIFile


notes = {'D2': 38, 'E2': 40, 'F2': 41, 'G2': 43, 'A2': 45, 'B2': 47, 'C3': 48, 'D3': 50, 'E3': 52, 'F3': 53, 'G3': 55,
         'A3': 57, 'B3': 59, 'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71, 'C5': 72, 'D5': 74,
         'E5': 76, 'F5': 77, 'G5': 79, 'A5': 81, 'B5': 83, 'C6': 84, 'D6': 86, 'E6': 88, 'F6': 89, 'G6': 91, 'A6': 93,
         'B6': 95, 'C7': 96, 'D7': 98, 'E7': 100, 'F7': 101, 'G7': 103, 'A7': 105, 'B7': 107, 'C8': 108, 'D8': 110,
         'E8': 112, 'F8': 113, 'G8': 115, 'A8': 117, 'B8': 119, 'C9': 120, 'D9': 122, 'E9': 124, 'F9': 125, 'G9': 127,
         'A9': 129, 'B9': 131, 'C10': 132, 'D10': 134, 'E10': 136, 'F10': 137, 'G10': 139, 'A10': 141, 'B10': 143}


def notify(sounds):
    global notes
    h = []
    for sound in sounds:
        h.append(notes[sound])
    return h

def build_midi(sounds, times):
    x = notify(sounds)
    track = 0
    channel = 0
    time = 0  # In beats
    duration = 1  # In beats
    tempo = 120  # In BPM
    volume = 100  # 0-127, as per the MIDI standard



    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, time, tempo)

    for i, pitch in enumerate(x):
        if times[i]==4:
            duration = 0.25
        elif times[i]==3:
            duration = 0.5
        elif times[i]==2:
            duration = 1
        elif times[i]==1:
            duration = 2
        elif times[i]==0:
            duration = 4

        MyMIDI.addNote(track, channel, pitch, time, duration, volume)

        time+=duration

    with open("fumo.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)


