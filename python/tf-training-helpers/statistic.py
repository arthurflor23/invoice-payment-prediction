import os
import re
import html
import string
import unicodedata
import editdistance
import statistics
from scipy.stats import wilcoxon


NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769),
                                                                                      chr(832), chr(833),
                                                                                      chr(2387), chr(5151),
                                                                                      chr(5152), chr(65344),
                                                                                      chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)


def text_standardize(txt):
    """Organize/add spaces around punctuation marks"""

    if txt is None:
        return ""

    txt = html.unescape(txt).replace("\\n", "").replace("\\t", "")

    txt = RE_RESERVED_CHAR_FILTER.sub("", txt)
    txt = RE_DASH_FILTER.sub("-", txt)
    txt = RE_APOSTROPHE_FILTER.sub("'", txt)
    txt = RE_LEFT_PARENTH_FILTER.sub("(", txt)
    txt = RE_RIGHT_PARENTH_FILTER.sub(")", txt)
    txt = RE_BASIC_CLEANER.sub("", txt)

    txt = txt.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    txt = NORMALIZE_WHITESPACE_REGEX.sub(" ", txt.strip())

    return txt

def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    cer, wer = [], []

    for (pd, gt) in zip(predicts, ground_truth):

        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd.lower()), list(gt.lower())
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.lower().split(), gt.lower().split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)

    return (cer_f, wer_f)


paths = ["bentham", "iam", "rimes", "saintgall"]
models = ["puigcerver", "bluche", "flor"]
modes = [["default", False, False], ["norm_accentuation", True, False], ["norm_punctuation", False, True], ["norm_accentuation_punctuation", True, True]]
dataset = dict()

for path in paths:
	dataset[path] = dict()

	for y, model in enumerate(models):
		root_path = os.path.join(path, model)
		os.makedirs(root_path, exist_ok=True)

		lines = open(os.path.join(path, f"predict_{model}.txt")).read().splitlines()
		ground_truth, predicts = [], []
		dataset[path][model] = dict()
		
		for line in lines:
			if len(line) == 0:
				continue

			if line.startswith("L:"):
				ground_truth.append(text_standardize(line[3:]))
			elif line.startswith("P:"):
				predicts.append(text_standardize(line[3:]))

		with open(os.path.join(root_path, f"resume.txt"), "w") as resume:
			if y != 0: print()

			print(f"{root_path.ljust(30)} \t\t CER \t st dev \t WER \t st dev")
			resume.write(f"\n{root_path.ljust(30)} \t\t CER \t st dev \t WER \t st dev\n")
			
			for i, mode in enumerate(modes):
				dataset[path][model][mode[0]] = dict()
			
				with open(os.path.join(root_path, f"{mode[0]}.txt"), "w") as f:
					batch_size = 8
					steps_done = 0
					steps = len(predicts) // batch_size

					f.write("CER;WER\n")

					while steps_done < steps:
						current_index = steps_done * batch_size
						until_index = current_index + batch_size
						steps_done += 1
						
						bt_pd = predicts[current_index:until_index]
						bt_gt = ground_truth[current_index:until_index]
						
						cer_f, wer_f = ocr_metrics(bt_pd, bt_gt, mode[1], mode[2])
						f.write(f"{cer_f};{wer_f}\n")

				lines = open(os.path.join(root_path, f"{mode[0]}.txt")).read().splitlines()

				cer = [float(line.split(";")[0]) for line in lines[1:]]
				cer_average =  (sum(cer) / len(cer)) * 100
				cer_st = (statistics.stdev(cer) / cer_average) * 100

				wer = [float(line.split(";")[1]) for line in lines[1:]]
				wer_average = (sum(wer) / len(wer)) * 100
				wer_st = (statistics.stdev(wer) / wer_average) * 100

				dataset[path][model][mode[0]]["cer"] = [cer_average, cer_st, cer]
				dataset[path][model][mode[0]]["wer"] = [wer_average, wer_st, wer]

				print(f"{i+1}. {mode[0].ljust(30)}   \t {cer_average:.2f}\t {cer_st:.2f}\t\t {wer_average:.2f}\t {wer_st:.2f}")
				resume.write(f"{i+1}. {mode[0].ljust(30)} \t {cer_average:.2f}\t {cer_st:.2f}\t\t {wer_average:.2f}\t {wer_st:.2f}\n")

	with open(os.path.join(root_path, f"resume.txt"), "a") as resume:
		for metric in ["cer", "wer"]:
			w, p = wilcoxon(dataset[path]["flor"]["default"][metric][2],
							dataset[path]["puigcerver"]["default"][metric][2],
							zero_method="wilcox",
							alternative="less")

			print(f"{metric.upper()} p-value: {p}")
			resume.write(f"\n{metric.upper()} p-value: {p}")
		print()

