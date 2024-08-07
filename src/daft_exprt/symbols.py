import string


# silence symbols and unknown word symbols used by MFA in ".TextGrid" files
MFA_SIL_WORD_SYMBOL = ''
MFA_SIL_PHONE_SYMBOLS = ['', 'sp', 'sil']
MFA_UNK_WORD_SYMBOL = '<unk>'
MFA_UNK_PHONE_SYMBOL = 'spn'

# silence symbols used in ".markers" files
# allows to only have 1 silence symbol instead of 3
SIL_WORD_SYMBOL = '<sil>'
SIL_PHONE_SYMBOL = 'SIL'

# PAD and EOS token
pad = '_'
eos = '~'

# whitespace character
whitespace = ' '

# punctuation to consider in input sentence
punctuation = ',.!?'
mandarin_punctuation = '，。！？'
# Arpabet stressed phonetic set
arpabet_stressed = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0',
                    'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1',
                    'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH',
                    'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
                    'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
mandarin = ['a1', 'a2', 'a3', 'a4', 'a5', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'b', 'c', 'ch', 'd', 'e1', 'e2', 'e3', 'e4', 'e5', 'ei1', 'ei2', 'ei3', 'ei4', 'f', 'g', 'h', 'i1', 'i2', 'i3', 'i4', 'i5', 'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 'iao1', 'iao2', 'iao3', 'iao4', 'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 'ii1', 'ii2', 'ii3', 'ii4', 'ii5', 'io1', 'io2', 'io3', 'io4', 'iou1', 'iou2', 'iou3', 'iou4', 'j', 'k', 'l', 'm', 'n', 'ng', 'o1', 'o2', 'o3', 'o4', 'o5', 'ou1', 'ou2', 'ou3', 'ou4', 'ou5', 'p', 'q', 'r', 's', 'sh', 't', 'u1', 'u2', 'u3', 'u4', 'u5', 'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'uai1', 'uai2', 'uai3', 'uai4', 'ue1', 'ue2', 'ue3', 'ue4', 'ue5', 'uei1', 'uei2', 'uei3', 'uei4', 'uei5', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'v1', 'v2', 'v3', 'v4', 'v5', 'va1', 'va2', 'va3', 'va4', 've1', 've2', 've3', 've4', 'x', 'z', 'zh']
# ascii letters
ascii = string.ascii_lowercase.upper() + string.ascii_lowercase

mandarin_ascii = ascii = string.ascii_lowercase.upper() + string.ascii_lowercase + string.digits

# symbols used by Daft-Exprt in english language
symbols_english = list(pad + eos + whitespace + punctuation) + arpabet_stressed
symbols_mandarin = list(pad + eos + whitespace + punctuation + mandarin_punctuation) + mandarin
