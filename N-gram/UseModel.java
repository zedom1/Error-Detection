
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.DocumentHelper;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.io.LmReaders;

public class UseModel {

	public static ArrayEncodedProbBackoffLm<String> model_raw;
	public static ArrayEncodedProbBackoffLm<String> model_pro;
	public static LmReaders readers;
	
	public static ArrayEncodedProbBackoffLm<String> getLm(boolean compress, String file){
		final File lmFile = new File(file);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = -100.0f;
		final ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), compress, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		return lm;
	}
	
	public static void loadModel() {
		readers = new LmReaders();
		model_raw = UseModel.getLm(false,"./SougouCorpus.arpa");
		model_pro = UseModel.getLm(false,"./model_pro.arpa");
	}
	
	public static ArrayList<List<String>>[] getList(String sentence){
		ArrayList<List<String>>[] list = new ArrayList[2];
		ArrayList< List<String> > listBi = new ArrayList<>();
		ArrayList< List<String> > listTri = new ArrayList<>();
		String[] words = sentence.split(" ");
		for(int i=0; i<words.length-1 ; i++) {
			List<String> temList = new ArrayList<>();
			temList.add(words[i]);
			temList.add(words[i+1]);
			listBi.add(temList);
			if( i < words.length-2 ) {
				List<String> tem = new ArrayList(temList);
				tem.add(words[i+2]);
				listTri.add(tem);	
			}
		}
		list[0]=listBi;
		list[1]=listTri;
		return list;
	}
	
	
	
	public static ArrayList< Float[] >[] getScore(String sentence) {

		ArrayList[] list = getList(sentence);
		
		ArrayList< Float[] >[] scorelist = new ArrayList[2];
		ArrayList< Float[] > scoreBi = new ArrayList<>();
		ArrayList< Float[] > scoreTri = new ArrayList<>();
		
		List< List< String > > list_bi = list[0];
		List< List< String > > list_tri = list[1];
		
		for(int i=0 ; i<list_bi.size(); i++) {
			Float[] scoreArray = new Float[2];
			scoreArray[0] = model_raw.getLogProb(list_bi.get(i));
			scoreArray[1] = model_pro.getLogProb(list_bi.get(i));
			scoreBi.add(scoreArray);
			
			if(i<list_bi.size()-1) {
				Float[] scoreArrayTri = new Float[2];
				scoreArrayTri[0] = model_raw.getLogProb(list_tri.get(i));
				scoreArrayTri[1] = model_pro.getLogProb(list_tri.get(i));
				scoreTri.add(scoreArrayTri);
			}
		}
		scorelist[0]=scoreBi;
		scorelist[1]=scoreTri;
		return scorelist;
	}
	
	public static void printScore(String sentence) {
		
		ArrayList[] list = getList(sentence);
		
		ArrayList< Float[] >[] scorelist = getScore(sentence);
		
		List< List< String > > list_bi = list[0];
		List< List< String > > list_tri = list[1];
		ArrayList< Float[] > scoreBi = scorelist[0];
		ArrayList< Float[] > scoreTri = scorelist[1];
		
		for(int i=0 ; i<list_bi.size(); i++) {
			System.out.println(list_bi.get(i)+" Raw Score = "+scoreBi.get(i)[0]+". Pro Score = "+scoreBi.get(i)[1]);
		}
		for(int i=0 ; i<list_tri.size(); i++) {
			System.out.println(list_tri.get(i)+" Raw Score = "+scoreTri.get(i)[0]+". Pro Score = "+scoreTri.get(i)[1]);
		}
		
	}
	
	public static void main(String[] args) throws IOException {
		
		UseModel reader = new UseModel();
		loadModel();
		Scanner input = new Scanner(System.in);
		
		while(true) {
			System.out.print("Input your sentence: ");
			String sentence = input.nextLine();
			printScore(sentence);
		}
		//	readFile();
		//getPatterns();
	}
	
	
	public static void readFile() throws IOException {
		
		/*
		处理搜狗语料库文件
		以GBK读入、处理XML、预处理、分词
		输出成txt文件用于建立n-gram模型
		
		文件以GBK编码打开
		文本预处理：
			1. 最开头增加root标签，最末尾增加root的结束标签
			2. 将所有 "&" 替换为 "&amp;"
		*/
		
		File dir = new File("./SougouCorpus");
		//System.out.println(dir.isDirectory());
		File tt = new File("./SougouCorpus_result.txt");
		PrintWriter pw = new PrintWriter(tt);
		int count =0;  
		for(File tem : dir.listFiles()) {
			if(count%10==0)
				System.out.println(count+"/"+dir.list().length);
			count++;
			String s = FileUtils.readFileToString(tem,"GBK");
			s = "<text>"+s+"</text>";
			s = s.replaceAll("&", "&amp;");
			SAXReader reader = new SAXReader();
			Document document = DocumentHelper.parseText(s);
			Element root = document.getRootElement();
			List nodes = root.elements("doc");
			for(Iterator it = nodes.iterator(); it.hasNext();) {
				Element elm = (Element) it.next();
				String ss = elm.element("content").getText().replaceAll("", "").trim();
				if(!ss.equals("")) {
					ss = BCConvert.qj2bj(ss).replaceAll("\\d+","*").replaceAll("http://.*com", "").replaceAll("(\\[\\*\\])+", "*").replaceAll("(\\*.\\*)+", "*").replaceAll("(\\* \\*)+", "");
					//List<Word> words = WordSegmenter.segWithStopWords(ss);
					pw.write(ss+"\n");
				}
			}
		}
		System.out.println("Finish!");
		pw.close();
		
	}
}
