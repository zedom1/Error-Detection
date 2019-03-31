from numpy import * 
from my import reader

counter = 0
resultList = []
def genPredict(result, test_path):
	global counter, resultList
	result = reshape(array(result),[-1,47])
	f = (open(test_path,"r").read().strip().split("\n"))
	i = 0
	while i < shape(result)[0]:
		length = len(f[counter+i].split())
		resultList.append(list(result[i])[:length])
		i+=1
	counter += i

def savePredict(index, test_path, config = None, describ = None):
	global resultList,counter

	f1 = open(test_path+"_ans").read().strip().split("\n")
	fresult = open("./report/report_"+str((test_path.split("/")[-1]).split("_")[-1])+".txt","a")
	print("=/*"*20)
	print(resultList)
	tp1=0
	tp05=0
	fp1 = 0
	fp05 = 0
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	pre = 0
	rec = 0
	acc = 0
	f1score = 0
	"""
	if len(f1)!= len(resultList):
		print("Invalid Result")
		print(resultList)
		print(len(f1))
		print(len(resultList))
		return
	"""
	for i in range(len(resultList)):
		ans = list(map(int,f1[i].split()))
		print(ans)
		predict = resultList[i]
		if min(predict) == 1:
			print("[-1]")
			if ans[0] == -1:
				tn += 1
			else:
				fn += 1
		else:
			predict = [a for a,x in enumerate(predict) if x==0]
			print(predict)
			if len(predict) > len(ans):
				fp1 += 1
			elif len(predict) == len(ans):
				if sorted(predict) == sorted(ans):
					tp1 += 1
				else:
					flag = 0
					for temind in predict:
						if temind in ans:
							flag = 1
							break
					if flag == 1:
						tp05 += 1
						fp05 += 1
					else:
						fp1 += 1
			else:
				flag = 1
				for temind in predict:
					if temind not in ans:
						flag = 0
				if flag == 1:
					tp05 += 1
				else:
					fp1 += 1
		print('-'*10)
	tp = tp1 + tp05*0.5
	fp = fp1 + fp05*0.5
	print("tp: %f fp: %f tn: %f fn: %f"%(tp,fp,tn,fn))
	acc = (tp+tn)/(tp+fp+tn+fn)
	if (tp+fp)>=0.1 and (tp+fn)>=0.1 and tp>0.1:
		pre = (tp/(tp+fp))
		rec = (tp/(tp+fn))
		f1score = 2*rec*pre/(rec+pre)


	ss =("========= Report ===========\n")
	ss+=("Index:%d\n"%(index))
	if describ is not None:
		ss += ("Description: %s\n"%(describ))
	if config is not None:
		ss += str(config)+"\n"
	ss+=("tp1:%d\n"%tp1)
	ss+=("tp05:%d\n"%tp05)
	ss+=("fp1:%d\n"%fp1)
	ss+=("fp05:%d\n"%fp05)
	ss+=("tn:%d\n"%tn)
	ss+=("fn:%d\n"%fn)
	ss+=("=========\n")
	if (tp+fp)>=0.1 and (tp+fn)>=0.1 and tp>0.1:
		ss+=("Precision\t=\t%f\n"%pre)
		ss+=("Recall\t\t=\t%f\n"%rec)
		ss+=("Accuracy\t=\t%f\n"%(acc))
		ss+=("F1\t\t=\t%f\n"%(f1score))
		ss+=("========= Report ===========\n")
	fresult.write(ss)
	fresult.close()

	counter = 0
	resultList = []
	return acc, f1score