import copy
def singleResult(f,f1,threshold):
	
	i = 0
	i1 = 0
	parti = 0
	flag = 0
	ls = []
	predictInd = []
	tp1 = 0 
	fp1 = 0
	tp05 = 0
	fp05 = 0

	tn = 0
	fn = 0
	while i<len(f):
		if f[i] == "===================":
			i += 1
			parti = 0
			ans = list(map(int,f1[i1].split()))
			if ans[0] != -1:
				ans = [a-1 for a in ans]
			i1 += 1
			if flag == 0:
				if ans[0] == -1:
					tn += 1
				else:
					fn += 1
			if len(predictInd)>=1:
				if len(predictInd) > len(ans):
					fp1 += 1
				elif len(predictInd) == len(ans):
					if sorted(predictInd) == sorted(ans):
						tp1 += 1
					else:
						flag = 0
						for temind in predictInd:
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
					for temind in predictInd:
						if temind not in ans:
							flag = 0
					if flag == 1:
						tp05 += 1
					else:
						fp1 += 1
			ls = []
			flag = 0
			predictInd = []
			continue
		line = f[i].split()
		if float(line[1]) < threshold:
			ls.append([line])
			predictInd.append(parti)
			flag = 1
		i += 1
		parti += 1
	pre = 0
	rec = 0
	f1score = 0
	acc = 0
	tp = tp1 + tp05*0.5
	fp = fp1 + fp05*0.5
	if (tp+fp)>=0.1 and (tp+fn)>=0.1 and tp>0.1:
		pre = (tp/(tp+fp))
		rec = (tp/(tp+fn))
		acc = (tp+tn)/(tp+fp+tn+fn)
		f1score = 2*rec*pre/(rec+pre)
	return threshold,tp1,tp05,fp1,fp05,tn,fn,tp,fp,pre,rec,acc,f1score

def generate(sentence, index, test_path, config, describ=None):
	f = sentence.strip().split("\n")
	f1 = open(test_path+"_ans").read().strip().split("\n")
	fresult = open("./report/report_"+str((test_path.split("/")[-1]).split("_")[-1])+".txt","a")

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
	threshold = 0
	for i in range(60,151):
		temresults = singleResult(f,f1,-i/10)
		if float(temresults[-1])>f1score:
			threshold,tp1,tp05,fp1,fp05,tn,fn,tp,fp,pre,rec,acc,f1score = temresults

	ss =("========= Report ===========\n")
	ss+=("Index:%d, threshold:%f\n"%(index,threshold))
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
	print("Accuracy: %.3f F1: %.3f"%(acc,f1score))
	fresult.write(ss)
	fresult.close()
