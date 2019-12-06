import os

from nltk.corpus import cmudict

import simpleaudio as sa
import argparse

import re
import string

import numpy as np

# NOTE: DO NOT CHANGE ANY OF THE EXISTING ARGUMENTS
parser = argparse.ArgumentParser(
    description='A basic text-to-speech app that synthesises an input phrase using diphone unit selection.')
parser.add_argument('--diphones', default="./diphones", help="Folder containing diphone wavs")
parser.add_argument('--play', '-p', action="store_true", default=False, help="Play the output audio")
parser.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                    default=None)
parser.add_argument('phrase', nargs=1, help="The phrase to be synthesised")

# Arguments for extensions
parser.add_argument('--spell', '-s', action="store_true", default=False,
                    help="Spell the phrase instead of pronouncing it")
parser.add_argument('--reverse', '-r', action="store_true", default=False,
                    help="Speak backwards")
parser.add_argument('--crossfade', '-c', action="store_true", default=False,
                    help="Enable slightly smoother concatenation by cross-fading between diphone units")
parser.add_argument('--volume', '-v', default=None, type=int,
                    help="An int between 0 and 100 representing the desired volume")

args = parser.parse_args()


class Synth:
    def __init__(self, wav_folder):
        self.diphones = {}
        self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        for root, dirs, files in os.walk(wav_folder, topdown=False):
            for file in files:
                # record the path of every wav files in diphones folder
                re_files = re.compile(r'^(.+)(.wav)$')
                # let name be the first parenthesized subgroup which for example return AA-AA for file aa-aa.wav
                name = re_files.match(str(file)).groups()[0].upper()
                self.diphones[name] = os.path.join(wav_folder, file)


def remove_seq_punctuations(phrase):
    # remove spaces
    phrase = phrase.strip()
    # delete the punctuation using default string.punctuation which is !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ for specific
    punctuations = string.punctuation
    # doing extension C and keep ,.?!
    punctuations = punctuations.replace(",", "")
    punctuations = punctuations.replace(".", "")
    punctuations = punctuations.replace("?", "")
    punctuations = punctuations.replace("!", "")
    result = phrase.translate(str.maketrans('', '', punctuations))
    # to lower case all sequences
    result = result.lower()

    return result


# remove numbers in the name and concatenate with "-" for each 2 sequence.
def normalise_diphone_seq(sequence):
    result = []
    sequence = re.sub(r'[0-9]', '', sequence)   # remove all numbers in sequence
    sub_seq = sequence.split()
    for i in range(0, len(sub_seq) - 1, 1):
        if sub_seq[i] not in ".,?!" and sub_seq[i + 1] not in ".,?!":  # if both adjacent elements are not .,?!
            result.append([sub_seq[i] + "-" + sub_seq[i + 1]][0])   # generate the aa-aa prefix and save to result.
        # add a [pau-(next element)] if there is a punctuation in middle of sentence.
        elif sub_seq[i] in ".,?!" and sub_seq[i + 1] not in ".,?!":
            result.append("PAU-" + sub_seq[i + 1])
        else:
            # keep the punctuations in the diphone sequence.
            result.append(sub_seq[i + 1])
    return result


# split the input string into list of character after normalization use for extension B
def synthesise(phrase):
    phrase = translate_date(phrase)
    phrase = remove_seq_punctuations(phrase)
    # remove all whitespace and separate phrase into characters.
    result = [phrase[i] for i in range(len(phrase)) if phrase[i] != " "]
    return result


# insert silence into audio.data use for extension C
def insert_silence(audio, time):
    silence = np.zeros(int(audio.rate * time), dtype=audio.nptype)
    audio.data = np.concatenate((audio.data, silence), axis=0)


def smoother(audio):  # Extension D
    # lower the amplitude at the end of one diphone down to 0.0 over 0.01 sec
    s1 = np.zeros(int(audio.rate * 0.01), dtype=audio.nptype)
    h1 = audio.data[-1] / len(s1)
    s1[0] = audio.data[-1] - h1
    # minus a step height for each iteration
    for i in range(1, len(s1)):
        s1[i] = s1[i-1] - h1
    audio.data = np.concatenate((audio.data, s1), axis=0)

    # add in the signal from the start of the next diphone
    # which is similarly tapered from 0.0 to normal amplitude over the same 0.01 sec period
    s2 = np.zeros(int(audio.rate * 0.01), dtype=audio.nptype)
    h2 = audio.data[0] / len(s2)
    s2[0] = h2
    # add a step height for each iteration
    for i in range(1, len(s2)):
        s2[i] = s2[i - 1] + h2

    audio.data = np.concatenate((s2, audio.data), axis=0)


def translate_date(phrase):
    # Defines the three different format of the dates that can be recognized
    re_date1 = re.compile(r"\d{2}\/\d{2}")   # regular expression for dd/mm
    re_date2 = re.compile(r'\d{2}\/\d{2}\/\d{2}')   # regular expression for dd/mm/yy
    re_date3 = re.compile(r'\d{2}\/\d{2}\/\d{4}')   # regular expression for dd/mm/yyyy

    days = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth",
        20: "twentieth",
        21: "twenty first",
        22: "twenty second",
        23: "twenty third",
        24: "twenty fourth",
        25: "twenty fifth",
        26: "twenty sixth",
        27: "twenty seventh",
        28: "twenty eighth",
        29: "twenty ninth",
        30: "thirtieth",
        31: "thirty first"
    }

    months = {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december"
    }

    # do at least the number of slashes times
    for i in range(phrase.count("/")):
        # filter from up to down is dd/mm/yyyy then dd/mm/yy finally dd/mm
        dates = re.findall(re_date3, phrase)
        if dates:
            # match dd/mm/yyyy
            for date in dates:
                # split into day month and year
                ddmmyyyy = date.split("/")
                day = int(ddmmyyyy[0])
                month = int(ddmmyyyy[1])
                year = int(ddmmyyyy[2])
                # as we only consider 1900-1999
                translated_date = months[month] + " " + days[day] + " nineteen " + translate_year(int(year-1900))
                phrase = phrase.replace(date, translated_date)
        else:
            dates = re.findall(re_date2, phrase)
            if dates:  # match dd/mm/yy
                for date in dates:
                    ddmmyy = date.split("/")
                    day = int(ddmmyy[0])
                    month = int(ddmmyy[1])
                    year = int(ddmmyy[2])
                    translated_date = months[month] + " " + days[day] + " nineteen " + translate_year(year)
                    phrase = phrase.replace(date, translated_date)
            else:
                dates = re.findall(re_date1, phrase)
                if dates:  # match dd/mm
                    for date in dates:
                        ddmm = date.split("/")
                        day = int(ddmm[0])
                        month = int(ddmm[1])
                        translated_date = months[month] + " " + days[day]
                        phrase = phrase.replace(date, translated_date)

    return phrase


def translate_year(yy):
    # helper function to help translate years into english
    first_ten_year = {
        0: "hundred",
        1: "O one",
        2: "O two",
        3: "O three",
        4: "O four",
        5: "O five",
        6: "O six",
        7: "O seven",
        8: "O eight",
        9: "O nine",
        10: "tens"
    }

    num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
                 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
                 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
                 19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty',
                 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty',
                 90: 'ninety', 0: 'zero'}

    if 0 <= yy <= 10:
        # the first 10 years are special.
        return first_ten_year[yy]
    else:
        try:
            return num2words[yy]
        except KeyError:
            try:
                # split 2 digits number into multiply of 10 and 1. for example 83 = 8*10+3
                return num2words[yy - yy % 10] + " " + num2words[yy % 10].lower()
            except KeyError:
                print('Number out of range')


class Utterance:
    def __init__(self, phrase):
        # initializer to pre-process the phrase
        translated_date = translate_date(phrase)
        self.phrase = translated_date
        self.phrase = remove_seq_punctuations(self.phrase)

    def get_phone_seq(self):
        # load the cmudict
        dip_seq = ["PAU"]  # initialise diphone sequence with a pause.
        cmudictionary = cmudict.dict()  # need every parts of word sequence matched with cmudict
        sequence = re.findall(r"[\w']+|[.,!?]", self.phrase)

        punctuations = string.punctuation
        # doing extension C and keep ,.?!
        punctuations = punctuations.replace(",", "")
        punctuations = punctuations.replace(".", "")
        punctuations = punctuations.replace("?", "")
        punctuations = punctuations.replace("!", "")

        for ss in sequence:
            try:
                if ss in ",.?!":
                    dip_seq.append("PAU")
                elif ss not in punctuations:
                    ss = cmudictionary[ss]
            # If it is not found in cmudict, throw an exception and exit the program
            except KeyError as inst:
                print("KeyError: \"" + inst.args[0] +
                      "\" is not in the cmudict")
                exit(1)
            dip_seq.append(" ".join(ss[0]))
        if sequence[-1] not in ".,?!":
            dip_seq.append("PAU")  # add a pause at end of the diphone sequence
        result = " ".join(dip_seq)
        return result


if __name__ == "__main__":
    # Initialize utt class, get the diphone sequence and os path
    utt = Utterance(args.phrase[0])
    diphone_seq = utt.get_phone_seq()
    diphone_synth = Synth(os.path.join(os.getcwd(), args.diphones))

    diphone_seq = normalise_diphone_seq(diphone_seq)
    # out is the Audio object which will become your output
    # you need to modify out.data to produce the correct synthesis

    out = sa.Audio(rate=16000)

    print(diphone_seq)

    # insert silence for comma and .?!
    for token in diphone_seq:
        d = sa.Audio(rate=16000)
        if token in ',':
            # 200ms which is 0.2s for comma
            insert_silence(out, 0.20)
        elif token in '.?!':
            insert_silence(out, 0.40)
        else:
            # load the wav file
            d.load(path=diphone_synth.diphones[token])
            # smooth the date using function smoother
            smoother(d)
        out.data = np.concatenate((out.data, d.data), axis=None)

    if args.play:
        # play the given sentence or word.
        out.play()
    elif args.outfile:
        # save as a wav file
        out.save(args.outfile)
    elif args.volume:
        # allow user to set the volume using rescale by a factor between 0 and 1.
        volume = args.volume / 100
        out.rescale(volume)
        # play the file after adjust the volume.
        out.play()
    elif args.spell:
        # spell the characters one by one
        out_spell = sa.Audio(rate=16000)
        chars = synthesise(args.phrase[0])
        chars = [chars[i] for i in range(len(chars)) if chars[i] not in ".,?!"]
        # print(chars)
        # using helper function synthesise to split into characters.
        for char in chars:
            # do the same process for each character
            utt_char = Utterance(char)
            char_seq = utt_char.get_phone_seq()
            char_seq = normalise_diphone_seq(char_seq)
            for token in char_seq:
                # concatenate into the total out_spell object
                d = sa.Audio(rate=16000)
                d.load(path=diphone_synth.diphones[token])
                out_spell.data = np.concatenate((out_spell.data, d.data), axis=None)
        out_spell.play()
