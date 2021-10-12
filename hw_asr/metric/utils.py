# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    t_text = target_text.split()
    p_text = predicted_text.split()
    if target_text == '' and predicted_text != '':
        return 1.0
    return editdistance.eval(t_text, p_text) / len(t_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '' and predicted_text != '':
        return 1.0
    return editdistance.eval(target_text, predicted_text) / len(target_text)
