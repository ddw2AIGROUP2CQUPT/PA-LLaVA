import sys
import re
import json,jsonlines
from collections import Counter

class VQAEval:
    def __init__(self, vqa_path, vqaRes_path, n=5):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        with open(vqa_path, 'r') as f:
            vqa = json.load(f)
        self.vqa = vqa
        try:
            with open(vqaRes_path, 'r') as f:
                data = json.load(f)
            self.vqaRes, qid = self.createIndex(data)
        except:
            with jsonlines.open(vqaRes_path, 'r') as f:
                data = list(f)
            self.vqaRes, qid = self.createIndex(data)
        self.params = {'question_id':qid}

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't",
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                             "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've",
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                             "Id've": "I'd've", "I'dve": "I'd've",
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                             "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                             "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't",
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                             "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                             "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                             "someone'dve": "someone'd've",
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                             "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've",
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                             "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't",
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                             "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've",
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                             "whos": "who's", "whove": "who've", "whyll": "why'll",
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                             "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've",
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                             "youd've": "you'd've", "you'dve": "you'd've",
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']
    def createIndex(self,data):
        vqa = {}
        for row in data:
            vqa[row['question_id']] = row
        
        return vqa, list(vqa.keys())

    def calculate_recall_rate_with_frequency(self, predicted_list, actual_list):

        predicted_counter = Counter(predicted_list)
        actual_counter = Counter(actual_list)
        

        matched_token_count = 0
        for token, actual_count in actual_counter.items():
            predicted_count = predicted_counter.get(token, 0)
            matched_token_count += min(predicted_count, actual_count)
        

        total_actual_tokens = sum(actual_counter.values())
        recall_rate = matched_token_count / total_actual_tokens if total_actual_tokens else 0
        
        return recall_rate


    def evaluate(self,recall=False, tokenizer=None):
        quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa[quesId]
            res[quesId] = self.vqaRes[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}

        print("computing accuracy")
        step = 0
        count_accurate = 0
        count_error = 0
        for quesId in quesIds:
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']

            resAns = res[quesId]['answer']
            resAns = pre_answer(resAns)

            gtAns = gts[quesId]['answer']
            gtAns = pre_answer(gtAns)
            
            if recall:
                gtstoken = tokenizer.encode(gtAns, add_special_tokens=False)
                restoken = tokenizer.encode(resAns, add_special_tokens=False)
                gtAcc = self.calculate_recall_rate_with_frequency(restoken,gtstoken)
            else:
                if gtAns == resAns:
                    count_accurate+=1
                    gtAcc = 1
                    
                else:
                    count_error += 1
                    gtAcc = 0.0

            

            accQA.append(gtAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(gtAcc)
            
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(gtAcc)
            self.setEvalQA(quesId, gtAcc)
            self.setEvalQuesType(quesId, quesType, gtAcc)
            self.setEvalAnsType(quesId, ansType, gtAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1
        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy")
        print('count_error == ', count_error)
        print('count_accurate == ', count_accurate)



    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accAnsType = accAnsType
        self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy['perQuestionType'] = {
            quesType: round(100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]), self.n) for quesType
            in accQuesType}
        self.accuracy['perAnswerType'] = {}
        try:
            self.accuracy['perAnswerType']['yes/no'] = round(100 * float(sum(accAnsType['yes/no'])) / len(accAnsType['yes/no']), self.n)
        except:
            pass
        try:
            self.accuracy['perAnswerType']['other'] = round(100 * float(sum(accAnsType['other']) + sum(accAnsType['number'])) / (len(accAnsType['other']) + len(accAnsType['number'])), self.n)
        except:
            print("No open datasets")

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), int(progress * 100),status)
        sys.stdout.write(text)
        sys.stdout.flush()


def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    return answer

