import os
import random

trainPath = "./corpus/total_cha.txt"
modelPath = "./model/model_big_e2/"
testPath  = "./test/test_total"


def genTestFile( num = 500, correct_rate = 0.3 ):

	word_to_id = eval(open("./cha_to_id.txt").read())
	id_to_word = {int(v):k for k,v in word_to_id.items()}

	testf = open(testPath,"w")
	ansf  = open(testPath+"_ans","w")
	trainf = open(trainPath).read().strip().split("\n")
	trainList = random.sample(trainf,num)
	trainList = [ [i for i in j.split() if i!="9173"] for j in trainList ]

	correctList = trainList[:int(num*correct_rate)]
	errorList = trainList[int(num*correct_rate):]

	for i in correctList:
		temline = [id_to_word[int(word)] for word in i]
		testf.write(' '.join(temline)+"\n")
		ansf.write("-1\n")

	for i in errorList:
		randInd = random.randint(1,len(i)-1)
		randWord = random.randint(0,len(id_to_word)-1)
		while randWord == i[randInd]:
			randWord = random.randint(0,len(id_to_word)-1)
		i[randInd] = randWord
		temline = [id_to_word[int(word)] for word in i]
		testf.write(' '.join(temline)+"\n")
		ansf.write(str(randInd)+"\n")
	testf.close()
	ansf.close()



if not os.path.exists(testPath):
	genTestFile()

os.system("python3 rnnlm.py --model test --test_path %s --save_path %s"%(testPath, modelPath))
