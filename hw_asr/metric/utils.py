# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    t_text = target_text.split()
    p_text = predicted_text.split()
    den = len(t_text)
    if target_text == '':
        den = 1
    return editdistance.eval(t_text, p_text) / den


def calc_wer(target_text, predicted_text) -> float:
    den = len(target_text)
    if target_text == '':
        den = 1
    return editdistance.eval(target_text, predicted_text) / den
