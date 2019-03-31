
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

public class ProcessData {

	public static ArrayList<Entity> testSentence( String sentence ,ArrayList< Float[] >[] score ) {
		
		/*
		 * 判断句子中是否有错误词语
		 * 
		 * 输入：
		 * 		以 " " 分割的完整待检测句子
		 * 		经过n-gram输出的score ArrayList数组（分别是2-gram和3-gram的分数），ArrayList里记录了句子中下标为1开始的词语的分数
		 * 
		 * 输出：
		 * 		错误词语的Entity ArrayList， 无错误词语则返回null
		 * 
		 * 过程：
		 * 		扫描score，若一个词语的2-gram和3-gram分数同时小于阙值（人工设置-5.5），则该词设为错误词语
		 * 		记录错误词语在句子中的Index，词语本身（用sentence split后得到），2-gram分数
		 * 
		 */
		
		String [] splitlist = sentence.split(" ");
		ArrayList<Entity> errorlist = new ArrayList<>();
		
		for(int i=1; i<splitlist.length; i++) {
			if (i>=2) {
				if( (score[0].get(i-1)[0]<-5.5&&score[0].get(i-1)[1]<-5.5) || ( score[1].get(i-2)[0]<-5.5&&score[1].get(i-2)[1]<-5.5 ) ) {
					Entity entity = new Entity();
					entity.name = splitlist[i];
					entity.index =i;
					entity.score[0] = score[0].get(i-1)[0]; 
					entity.score[1] = score[0].get(i-1)[1]; 
					entity.score[2] = score[1].get(i-2)[1];
					entity.score[3] = score[1].get(i-2)[1];
					errorlist.add(entity);
				}
			}
			else
				if( (score[0].get(i-1)[0]<-5.5&&score[0].get(i-1)[1]<-5.5) ) {
					Entity entity = new Entity();
					entity.name = splitlist[i];
					entity.index =i;
					entity.score[0] = score[0].get(i-1)[0]; 
					entity.score[1] = score[0].get(i-1)[1]; 
					errorlist.add(entity);
				}
		}
		return errorlist.size()==0?null:errorlist;
	}
	
	public static void get_txt() throws FileNotFoundException {
		
		/*
		 * 使用n-gram判断句子分数生成txt文件（便于人工观察错误）
		 * 
		 * 输入：
		 * 		分好词、做好预处理的待检测的txt文件，（大句子以逗号分割成小句子，以小句子检测错误）
		 * 
		 * 输出：
		 * 		词典列表的txt文件，包含错误词语的下标、词语本身、score
		 * 		
		 */
		
		File file = new File("./raw_processed.txt");
		File output = new File("./result.txt");
		Scanner input = new Scanner(file);
		PrintWriter pw = new PrintWriter(output);
		
		Segment seg = HanLP.newSegment().enablePartOfSpeechTagging(false);
		int co = 0;
		// JSON 内部
		while(input.hasNext()) {
			if(co%100==0)
				System.out.println(co);
			co++;
			String tem = input.nextLine();
			
			String []tem1 = tem.trim().split(" ");
			int id = Integer.parseInt(tem1[0]);
			int count = Integer.parseInt(tem1[1]);
			// 每个大句子(需要根据逗号进行拆分)
			// count表明这个id有多少个大句子
			while((count--)>0) {
				
				// 解析每个以逗号分割的小句子
				ArrayList<String> sentenceList = new ArrayList<String>();
				while(true) {
					String sentence = input.nextLine();
					if( sentence.equals("-----")  ) 
						break;
					List<Term> termlist = seg.seg(tem);
					tem = "";
					for (Term term : termlist)
						tem = tem+" "+term.word;
					tem = BCConvert.qj2bj(sentence).replaceAll("\\d+","*").replaceAll("(\\[\\*\\])+", "*").replaceAll("(\\*.\\*)+", "*").replaceAll("(\\* \\*)+", "*").replaceAll("(\\*)+", "*");
					sentenceList.add(tem);
				}
				for(int j=0; j<sentenceList.size(); j++){
					// 错误词典 序列
					
					ArrayList< Float[] >[] score = UseModel.getScore(sentenceList.get(j));
					ArrayList<Entity> a = testSentence(sentenceList.get(j), score);
					
					// 输出错误单词词典	
					if(a!=null) {
						pw.write(sentenceList.get(j)+"\n");
						for( int i=0; i<a.size() ; i++) {
							pw.write("{"+"");
							pw.write("\"index\":"+a.get(i).index+",");
							pw.write("\"item\":\""+a.get(i).name+"\",");
							pw.write("\"score1\":"+a.get(i).score[0]+",");
							pw.write("\"score2\":"+a.get(i).score[1]+"");
							pw.write("}"+"\n");
						}
						pw.write("----------\n");
					}
					
				}

			}
		}
		System.out.println("Finish!");
		pw.close();
		input.close();
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		//UseModel.loadModel();
		get_txt();
	}

}
