import torch

ARPABET_PHONEMES = [
    "PAD", "UNK", "SIL", "SP", 
    "AA","AE","AH","AO","AW","AX","AXR","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","B","CH","D","DH","DX","F","G","HH","JH","K","L","M","N","NG","P","R","S","SH","T","TH","V","W","Y","Z","ZH"
]

PHONEME_TO_ID = {ph: i for i, ph in enumerate(ARPABET_PHONEMES)}
ID_TO_PHONEME = {i: ph for i, ph in enumerate(ARPABET_PHONEMES)}

def normalize_phoneme(ph):
    ph = ph.upper()
    ph = ''.join([c for c in ph if not c.isdigit()])
    if ph == "AXR":
        return "ER"
    return ph

def get_phoneme_id(ph):
    ph = normalize_phoneme(ph)
    return PHONEME_TO_ID.get(ph, PHONEME_TO_ID["UNK"])