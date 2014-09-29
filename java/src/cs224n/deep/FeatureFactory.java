package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;
                BufferedReader rd = new BufferedReader(new FileReader(vecFilename));
                
                //vectors are n=50 dimensional
		allVecs = new SimpleMatrix(numToWord.size(), 50);
                int row = 0;
                for (String ln = rd.readLine(); ln != null; ln = rd.readLine()) {
                  String[] lineNumbers = ln.trim().split("\\s+");
                  for (int col = 0; col < lineNumbers.length; col++) allVecs.set(row, col, Double.parseDouble(lineNumbers[col]));
                  row++;
                }
                return allVecs;
	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
	  BufferedReader rd = new BufferedReader(new FileReader(vocabFilename));
          
          int num = 0;
          for (String word = rd.readLine(); word != null; word = rd.readLine()) {
            wordToNum.put(word, num);
            numToWord.put(num, word);
            num++;
          }	
          return wordToNum;
	}
 








}
