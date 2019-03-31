

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.io.ArpaLmReader;
import edu.berkeley.nlp.lm.io.LmReaders;
import edu.berkeley.nlp.lm.util.Logger;

public class BuildModel {

	public static void usage() {
		System.err.println("Usage: <lmOrder> <ARPA lm output file> <textfiles>*");
		System.exit(1);
	}
	
	public void makelml(String [] argv) {
		// 输入格式：
		// int + 输出文件名 + 一系列输入文件名
		if(argv.length<2) {
			usage();
		}
		final int lmOrder = Integer.parseInt(argv[0]);
		final String outputFile = argv[1];
		final List<String> inputFiles = new ArrayList<String>();
		for (int i=2; i<argv.length ; ++i)
			inputFiles.add(argv[i]);
		if(inputFiles.isEmpty())
			inputFiles.add("-");
		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		Logger.startTrack("Reading text files " + inputFiles + " and writing to file " + outputFile );
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		wordIndexer.setStartSymbol(ArpaLmReader.START_SYMBOL);
		wordIndexer.setEndSymbol(ArpaLmReader.END_SYMBOL);
		wordIndexer.setUnkSymbol(ArpaLmReader.UNK_SYMBOL);
		LmReaders.createKneserNeyLmFromTextFiles(inputFiles, wordIndexer, lmOrder, new File(outputFile), new ConfigOptions());
		Logger.endTrack();
		
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		BuildModel t = new BuildModel();
		String inputFile = "./SougouCorpus_result.txt";
		String outputFile = "./SougouCorpus.arpa";
		String s[] = {"3",outputFile, inputFile};
		t.makelml(s);
	}

}