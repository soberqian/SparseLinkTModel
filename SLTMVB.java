package qian.lda.model;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import qian.lda.util.Sampler;


/**
 * Sparse Topic-Level Model--CVB
 * 
 * @author: qianyang
 * @email qy20115549@126.com
 */
public class SLTMVB {

	int[][] documents;
	int[][] entities;
	int V;
	int E;
	int K;
	//hyperparams
	double alpha0 ;  //������ alpha0 = 0.1
	double alpha1 ;  //������alpha1 = 1E-12
	double beta;
	double epsilon0;  //beta�ֲ���Ӧ�Ĳ���
	double epsilon1; //beta�ֲ���Ӧ�Ĳ���
	double beta_bar;
	//variational parameters
	double[][] b_mk; //�ĵ�����ѡ����
	double a_sum[];  //�ĵ�����ĸ���(����)
	double[][] nmk; //�ĵ�d������k���ɵĵ�����Ŀ(����)nmk+cmk
	double[] nm; //�ĵ�d�����Ĵ���Ŀ(����)nm+cm
	double[][] nkw; //����k�����ĵ���w����Ŀ(����) K*V
	double[][] nke; //����k������ʵ��e����Ŀ(����) K*E
	double[] nkw_sum; //����k�������ܵ�����Ŀ(����)
	double[] nke_sum; //����k��������ʵ������(����)
	double[][][] gamma_word; 
	double[][][] gamma_entity; 
	JDKRandomGenerator rand; //�����������
	int iterations;  //��������
	public SLTMVB(int[][] documents, int[][] entities, int V, int E) {

		this.documents = documents;
		this.entities = entities;
		this.V = V;
		this.E = E;
	}
	//��ʼ��
	public void initialState() {
		rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		//�����ֲ�����
		BetaDistribution betaDist = new BetaDistribution(rand, epsilon1 , epsilon0);
		//�ĵ�����
		int D = documents.length;
		//variational parameters
		b_mk = new double[D][K];
		a_sum = new double[D]; 
		nmk = new double[D][K];
		nm = new double[D];
		nkw = new double[K][V];
		nke = new double[K][E];
		nkw_sum = new double[K];
		nke_sum = new double[K];
		gamma_word = new double[D][][]; 
		gamma_entity = new double[D][][]; 
		//ѭ��ÿƪ�ĵ�
		for (int d = 0; d < D; d++) {
			// �ĵ���������
			int Nd = documents[d].length;
			//�ĵ�ʵ������
			int Ed = entities[d].length;
			gamma_word[d] = new double[Nd][K];
			gamma_entity[d] = new double[Ed][K];
		}
		//�����ʼ��
		for(int d = 0; d < D; d ++) {
			double[] theta;
			theta = Sampler.getDirichletSample(K, 0.1);
			double[] b_sigma = new double[K];
			double beta_norm = 0;
			for(int k = 0; k < K; k ++) {
				b_sigma[k] = 0.5;
				b_mk[d][k] = betaDist.sample();
				beta_norm += b_mk[d][k];
			}
			for (int k = 0; k < K; k ++) {
				b_mk[d][k] /= beta_norm;
				a_sum[d] += b_mk[d][k];
			}
			for(int n = 0; n < documents[d].length; n ++) {
				gamma_word[d][n] = Sampler.getGaussianSample(K, theta, b_sigma);
				double gamma_norm = 0;
				for(int k = 0; k < K; k ++) {
					gamma_norm += Math.exp(gamma_word[d][n][k]);
				}

				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] = Math.exp(gamma_word[d][n][k]) / gamma_norm;
					nkw_sum[k] += gamma_word[d][n][k];
					nmk[d][k] += gamma_word[d][n][k];
					nkw[k][documents[d][n]] += gamma_word[d][n][k];
					nm[d] += gamma_word[d][n][k];
				}
			}
			for(int n = 0; n < entities[d].length; n ++) {
				gamma_entity[d][n] = Sampler.getGaussianSample(K, theta, b_sigma);
				double gamma_norm = 0;
				for(int k = 0; k < K; k ++) {
					gamma_norm += Math.exp(gamma_entity[d][n][k]);
				}
				for(int k = 0; k < K; k ++) {
					gamma_entity[d][n][k] = Math.exp(gamma_entity[d][n][k]) / gamma_norm;
					nke_sum[k] += gamma_entity[d][n][k];
					nmk[d][k] += gamma_entity[d][n][k];
					nke[k][entities[d][n]] += gamma_entity[d][n][k];
					nm[d] += gamma_entity[d][n][k];
				}
			}
		}
	}
	//CVB0
	public void runModel(int K, double alpha0, double alpha1, double epsilon0, double epsilon1, double beta, double beta_bar, int iterations) throws Exception {
		this.K = K;   
		this.alpha0 = alpha0;
		this.alpha1 = alpha1;
		this.epsilon0 = epsilon0;
		this.epsilon1 = epsilon1;
		this.beta = beta;
		this.beta_bar = beta_bar;
		this.iterations = iterations;
		initialState();
		System.out.println("initial over......");
		for (int i = 0; i < this.iterations; i++) {
			System.out.println("iteration.................... " + i);
			iterateCVB0Update();  //ִ�б���ƶ�
		}
	}
	public void iterateCVB0Update(){
		int D = documents.length;
		//update b_mk
		for(int d = 0; d < D; d ++) {
			double[] prev_b = new double[K];
			for(int k = 0; k < K; k ++) {
				prev_b[k] = b_mk[d][k];
				double log_b1 = logOn2(epsilon0 + count_Am(d,k)) + 
						logOn2Gamma(nmk[d][k] + alpha0 + alpha1) + log2betaf(alpha0 + alpha0*count_Am(d,k) + K*alpha1, nm[d] + alpha0*count_Am(d,k) + K*alpha1);
				double log_b0 = logOn2(epsilon1 + K - 1.0 - count_Am(d,k)) +
						logOn2Gamma(alpha0 + alpha1) + log2betaf(alpha0*count_Am(d,k) + K*alpha1, nm[d] + alpha0*count_Am(d,k) + alpha0 + K*alpha1);
				if (exponential2(log_b1) > 1024) {
					b_mk[d][k] = Double.MAX_VALUE/(Double.MAX_VALUE + exponential2(log_b0));
				}else {
					b_mk[d][k] = exponential2(log_b1)/(exponential2(log_b1) + exponential2(log_b0));
				}
			}
			for(int k = 0; k < K; k ++) {
//				b_mk[d][k] /= norm_b;
				a_sum[d] +=  b_mk[d][k] - prev_b[k]; 
			}
		}
		//update gamma_word and gamma_entity
		for(int d = 0; d < D; d ++) {
			for(int n = 0; n < documents[d].length; n ++) {
				double norm_w = 0;
				double[] prev_gamma_w = new double[K];
				for(int k = 0; k < K; k ++) {
					prev_gamma_w[k] = gamma_word[d][n][k];
					gamma_word[d][n][k] = (mean_count_gamma_w(d, n, k, 0, d) +  + b_mk[d][k]*alpha0 + alpha1)*
							(beta + mean_count_gamma_w(d, n, k, documents[d][n], -1))/(V * beta + mean_count_gamma_w(d, n, k, 0, -1));
					norm_w += gamma_word[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] /= norm_w;
					//update
					nkw_sum[k] += gamma_word[d][n][k] - prev_gamma_w[k];
					nmk[d][k] += gamma_word[d][n][k] - prev_gamma_w[k];
					nkw[k][documents[d][n]] += gamma_word[d][n][k] - prev_gamma_w[k];
					nm[d] += gamma_word[d][n][k] - prev_gamma_w[k];
				}
			}
			for (int n = 0; n < entities[d].length; n ++) {
				double norm_e = 0;
				double[] prev_gamma_e = new double[K];
				for(int k = 0; k < K; k ++) {
					prev_gamma_e[k] = gamma_entity[d][n][k];
					gamma_entity[d][n][k] = (mean_count_gamma_e(d, n, k, 0, d) +  + b_mk[d][k]*alpha0 + alpha1)*
							(beta_bar + mean_count_gamma_e(d, n, k, entities[d][n], -1))/(E * beta_bar + mean_count_gamma_e(d, n, k, 0, -1));
					norm_e += gamma_entity[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_entity[d][n][k] /= norm_e;
					//update
					nke_sum[k] += gamma_entity[d][n][k] - prev_gamma_e[k];
					nmk[d][k] += gamma_entity[d][n][k] - prev_gamma_e[k];
					nke[k][entities[d][n]] += gamma_entity[d][n][k] - prev_gamma_e[k];
					nm[d] += gamma_entity[d][n][k] - prev_gamma_e[k];
				}
			}
		}
	}
	//��ض�������
	private static double logOn2Gamma(double value) {
		return com.aliasi.util.Math.log2Gamma(value);
	}
	private static double logOn2(double value) {
		return com.aliasi.util.Math.log2(value);
	}
	public static double log2betaf(double a,double b){
		double beta = logOn2Gamma(a)+ logOn2Gamma(b)-logOn2Gamma(a+b);
		return beta;
	}
	public static double exponential2(double a){
		return java.lang.Math.pow(2.0, a);
	}
	//a_sumͳ��
	private double count_Am(int d, int k) {
		return a_sum[d] - b_mk[d][k];
	}
	private double mean_count_gamma_w(int ex_d, int ex_n, int k, int wsdn, int doc) {
		if(wsdn == 0 && doc == -1)
			return nkw_sum[k] - gamma_word[ex_d][ex_n][k];
		else if(doc == -1)
			return nkw[k][wsdn] - gamma_word[ex_d][ex_n][k];
		else
			return nmk[doc][k] - gamma_word[ex_d][ex_n][k];
	}
	private double mean_count_gamma_e(int ex_d, int ex_n, int k, int wsdn, int doc) {
		if(wsdn == 0 && doc == -1)
			return nke_sum[k] - gamma_entity[ex_d][ex_n][k];
		else if(doc == -1)
			return nke[k][wsdn] - gamma_entity[ex_d][ex_n][k];
		else
			return nmk[doc][k] - gamma_entity[ex_d][ex_n][k];
	}
	
	//����Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[documents.length][K];
		for (int d = 0; d < documents.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (nmk[d][k] + + b_mk[d][k]*alpha0 + alpha1) / (nm[d] + K * alpha0);
			}
		}
		return theta;
	}
	//����Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nkw_sum[k] + V * beta);
			}
		}
		return phi;
	}
	//����PhiBar
	public double[][] estimatePhiBar() {
		double[][] phi_bar = new double[K][E];
		for (int k = 0; k < K; k++) {
			for (int e = 0; e < E; e++) {
				phi_bar[k][e] = (nke[k][e] + beta_bar) / (nke_sum[k] + E * beta_bar);
			}
		}
		return phi_bar;
	}
}
