package representative;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import com.google.common.collect.Maps;

//ͳ����������жȺ�ÿ��������һ����ֵ�µĴʵ���Ŀ
public class RepCount {
	public static void main(String[] args) throws IOException {
		String inputfile_p = "";
		String inputfile_word = "";
		LinkedHashMap<String,Double> popularty = populartySubmarket(inputfile_p);
		LinkedHashMap<String,LinkedHashMap<String,Double>> tword = readTWord(inputfile_word);
		//������
		double summary[] =new double[66];
		for (String key : popularty.keySet()) {
			LinkedHashMap<String,Double> wordpro = tword.get("Topic"+key);
			int count = 0;
			for (String key1 : wordpro.keySet()) {
				if (wordpro.get(key1) > 0.01) {
					count++;
				}
			}
			summary[Integer.parseInt(key)] = count;
			System.out.println(key + "\t" + popularty.get(key) + "\t" + count);
		}
		//���ֵ�뷽��
		System.out.println("��ֵΪ:" + getAverage(summary));
		System.out.println("����Ϊ:" + getStandardDeviation(summary));

	}
	//��ÿ����������ж�
	private static LinkedHashMap<String,Double> populartySubmarket(String inputfile) throws IOException{
		LinkedHashMap<String,Double> submarketPop = new LinkedHashMap<>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputfile)));
		String lineTxt;
		while( (lineTxt=reader.readLine()) != null ){
			submarketPop.put(lineTxt.split("\t")[0],
					Double.valueOf(lineTxt.split("\t")[1]));
		}
		reader.close();
		return submarketPop;
	}
	//��ÿ������Ĵʸ���
	private static LinkedHashMap<String,LinkedHashMap<String,Double>> readTWord(String inputfile) throws IOException{
		LinkedHashMap<String,LinkedHashMap<String,Double>> tWordProb = new LinkedHashMap<>();
		LinkedHashMap<String,Double> wordProb = new LinkedHashMap<>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputfile)));
		String lineTxt;
		String topicName = "";
		int i = 0;
		while( (lineTxt=reader.readLine()) != null ){
			if( lineTxt.startsWith("Topic")){
				if(!wordProb.isEmpty()){
					if( i==0 ){
						tWordProb.put(lineTxt, wordProb);
						wordProb = Maps.newLinkedHashMap();
					}else{
						tWordProb.put(topicName, wordProb);
						wordProb = Maps.newLinkedHashMap();
					}
				}
				topicName = lineTxt;

			}else{
				wordProb.put(lineTxt.split(" ")[0].replace("\t", ""),
						Double.valueOf(lineTxt.split(" ")[1]));
			}
			++i;
		}
		tWordProb.put(topicName, wordProb);
		reader.close();
		return tWordProb;
	}
	//��׼���=sqrt(s^2)
	public static double getStandardDeviation(double[] x) { 
		int m=x.length;
		double sum=0;
		for(int i=0;i<m;i++){//���
			sum+=x[i];
		}
		double dAve=sum/m;//��ƽ��ֵ
		double dVar=0;
		for(int i=0;i<m;i++){//�󷽲�
			dVar+=(x[i]-dAve)*(x[i]-dAve);
		}
		return Math.sqrt(dVar/m);	
	}
	//��ƽ��ֵ
	public static double getAverage(double[] x){
		double sum = 0;
		int num = x.length;
		for(int i = 0;i < num;i++){
			sum += x[i];
		}
		return (double)(1.0*sum / num);
	}
}
