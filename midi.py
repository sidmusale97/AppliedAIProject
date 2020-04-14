from mido import Message, MidiFile, MidiTrack
from predict import predict

notes_len_dict = {'eigth': 1,'quarter': 2, 'half': 4, 'whole': 8}


def createMidi(notes):
    new_song = MidiFile(ticks_per_beat=2, type=0)
    track = MidiTrack()
    new_song.tracks.append(track)

    for note in notes:
        len_str, note_num = note

        note_len = notes_len_dict[len_str]
        print('Note: %d Length: %d' % (note_num, note_len))
        msg = Message('note_on', note=note_num, time=note_len)
        track.append(msg)

    new_song.save('song.mid')

notes = predict('Capture.png')
createMidi(notes)


'''
new_song = MidiFile(ticks_per_beat=1)
track = MidiTrack()
new_song.tracks.append(track)

msg = Message('note_on', note=60, time=2)
msg2 = Message('note_on', note=60, time=4)
track.append(Message('program_change', program=1, time=0))
track.append(msg)
track.append(msg2)
track.append(msg)

new_song.save('song.mid')
'''