package qian.sparsel.model;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.special.Gamma;

/**
 * SparseLinkTopicModel GibbsSampling
 * 
 * @author: qianyang
 * @email qy20115549@126.com
 */
public class SparseLinkToModel {

	int[][] documents;

	int[][] entities;

	int V;

	int E;

	int K;

//	double alpha;

	double alpha0 ;  //������ alpha0 = 1E-12

	double alpha1 ;  //������alpha1 = 0.1

	double beta;

	double gamma0;  //beta�ֲ���Ӧ�Ĳ���

	double gamma1; //beta�ֲ���Ӧ�Ĳ���

	double beta_bar;

	int[][] z;

	int[][] z_bar;

	int[][] nw;

	int[][] ne;

	int[][] nd;

	int[] nwsum;

	int[] nesum;

	int[] ndsum;

	double pi_a[];  //pi����

	boolean a[][]; //�ĵ�����ѡ����

	int a_sum[];  //�ĵ�����ĸ���

	JDKRandomGenerator rand; //�����������

	int iterations;

	public SparseLinkToModel(int[][] documents, int[][] entities, int V, int E) {

		this.documents = documents;
		this.entities = entities;
		this.V = V;
		this.E = E;
	}
	//��ʼ������
	public void initialState() {
		rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		//�����ֲ�����
		BetaDistribution betaDist = new BetaDistribution(rand, gamma1 , gamma0);
		//�ĵ�����
		int D = documents.length;
		//�ĵ�d������k���ɵĵ�����Ŀ
		nd = new int[D][K];
		//�ĵ�m�Ƿ��������k
		a = new boolean[D][K]; 
		//�ĵ�m�������������
		a_sum = new int[D]; 
		//pi����
		pi_a = new double[D];  
		//ÿƪ�ĵ����������е�������
		ndsum = new int[D];
		//����k�е���v����Ŀ
		nw = new int[V][K];
		//����k��Ӧ�ĵ�������
		nwsum = new int[K];
		//����k�ж�Ӧ��ʵ��e����Ŀ
		ne = new int[E][K];
		//����k��ʵ�������
		nesum = new int[K];
		//ÿƪ�ĵ����ʶ�Ӧ������
		z = new int[D][];
		//ÿƪ�ĵ���Ӧ��ʵ���Ӧ������
		z_bar = new int[D][];
		//��ÿһ���ĵ���ȡpi�������ȡֵ
		for (int d = 0; d < D; d++) {

			pi_a[d] = betaDist.sample();
			//�տ�ʼ��ʼ���ĵ�������������
			for (int k = 0; k < K; k++) {
				a[d][k] = true;
			}
			//�ĵ�������������Ŀ�ܺ�
			a_sum[d] = K;
		}
		//ѭ��ÿһƪ�ĵ�
		for (int d = 0; d < D; d++) {

			// words
			int Nd = documents[d].length;
			z[d] = new int[Nd];
			//ѭ��ÿһ������
			for (int n = 0; n < Nd; n++) {
				//�����ֵ����
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
				//����ͳ��
				updateCount(d, topic, documents[d][n], +1);
			}

			// ��ȡ�ĵ���Ӧ������ʵ�����
			int Ed = entities[d].length;
			//�ĵ�ʵ���Ӧ������
			z_bar[d] = new int[Ed];
			//ѭ���ĵ�������ʵ��
			for (int m = 0; m < Ed; m++) {
				//�����ֵ����
				int topic = (int) (Math.random() * K);
				z_bar[d][m] = topic;
				//����ͳ��
				updateEntityCount(d, topic, entities[d][m], +1);
			}

		}

	}

	public void markovChain(int K, double alpha0, double alpha1, double gamma0, double gamma1, double beta, double beta_bar, int iterations) {

		this.K = K;   
		this.alpha0 = alpha0;
		this.alpha1 = alpha1;
		this.gamma0 = gamma0;
		this.gamma1 = gamma1;
		this.beta = beta;
		this.beta_bar = beta_bar;
		this.iterations = iterations;
		//��ʼ��--��������
		initialState();
		//��������
		for (int i = 0; i < this.iterations; i++) {

			System.out.println("iteration : " + i);
			gibbs();  //ִ��gibbs����
			//ϡ��ȼ���
			if(i % 1 == 0) {
				//��ȡ��Ԫ����
				sampleBinaryAMatrix();
			}
		}
	}
	//gibbs����
	public void gibbs() {
		//ѭ�������ĵ�
		for (int d = 0; d < documents.length; d++) {
			//ѭ�����е���
			for (int n = 0; n < z[d].length; n++) {
				//
				int topic = sampleFullConditional(d, n);
				z[d][n] = topic;

			}
			// ѭ������ʵ��
			for (int m = 0; m < z_bar[d].length; m++) {

				int topic = sampleFullConditionalEntity(d, m);
				z_bar[d][m] = topic;

			}
		}
	}
	//��ȡ���ʵ�����;�������Ϊ�ĵ�d�Լ��ĵ�d�еĵ���n
	int sampleFullConditional(int d, int n) {
		//��ȡԭ��Ӧ������
		int topic = z[d][n];
		//������Ŀ,��1
		updateCount(d, topic, documents[d][n], -1);
		//����
		double[] p = new double[K];
		//ѭ��ÿ������
		for (int k = 0; k < K; k++) {
			int x = a[d][k] ? 1 : 0;
			p[k] = (nd[d][k] + x*alpha1 + alpha0) / (ndsum[d] + K * alpha0) * (nw[documents[d][n]][k] + beta)
					/ (nwsum[k] + V * beta);
		}
		//���̶ĳ�ȡ������
		topic = sample(p);
		//����ͳ��
		updateCount(d, topic, documents[d][n], +1);
		//��������
		return topic;

	}
	//��ȡʵ�������
	int sampleFullConditionalEntity(int d, int m) {

		int topic = z_bar[d][m];

		updateEntityCount(d, topic, entities[d][m], -1);

		double[] p = new double[K];

		for (int k = 0; k < K; k++) {
			int x = a[d][k] ? 1 : 0;
			p[k] = (nd[d][k] + x*alpha1 + alpha0) / (ndsum[d] + K * alpha0) * (ne[entities[d][m]][k] + beta_bar)
					/ (nesum[k] + E * beta_bar);
		}
		topic = sample(p);

		updateEntityCount(d, topic, entities[d][m], +1);

		return topic;

	}
	//���̶�
	int sample(double[] p) {

		int topic = 0;
		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[p.length - 1];
		for (int t = 0; t < p.length; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}
		return topic;
	}

	//����ͳ��
	void updateCount(int d, int topic, int word, int flag) {
		//�ĵ� d�е�����topic��Ӧ�ĵ�����Ŀ��1
		nd[d][topic] += flag;
		//�ĵ�d
		ndsum[d] += flag;
		//����topic��Ӧ�ĵ���word������1
		nw[word][topic] += flag;
		//����topic��Ӧ�ĵ���������1
		nwsum[topic] += flag;
	}
	//����ͳ��
	void updateEntityCount(int d, int topic, int entity, int flag) {
		//����topic��ʵ��entity����Ŀ����1
		ne[entity][topic] += flag;
		//����topic��Ӧ���ܵ�ʵ����Ŀ��1
		nesum[topic] += flag;
		//�ĵ� d�е�����topic��Ӧ�ĵ�����Ŀ��1
		nd[d][topic] += flag;
		//�ĵ���Ӧ�ĵ����Լ�ʵ����ܸ���
		ndsum[d] += flag;
	}
	//��ȡ�ĵ�����ѡ����
	public void sampleBinaryAMatrix() {
		int GIBBS_ITER = 1;
		//�ĵ�ѡ������ĸ���
		a_sum = new int[documents.length];
		//ѭ��ÿһƪ�ĵ�
		for (int m = 0; m != documents.length; m++) {
			//ѭ��ÿ������
			for (int k = 0; k != K; k++) {
				//�ж��ĵ��Ƿ��Ѿ��и�������,�������Ϊtrue
				a[m][k] = (nd[m][k]) > 0;
				//�ĵ�m�������������+1
				a_sum[m] += a[m][k] ? 1 : 0;
			}
		}
		//
		double log_diff, ratio, p;
		for (int iter = 0; iter != GIBBS_ITER; iter++) {
			for (int m = 0; m != documents.length; m++) {
				for (int k = 0; k != K; k++) {
					if (a[m][k] && (nd[m][k])  == 0) {
						log_diff = Gamma.logGamma(a_sum[m]*alpha1 + K*alpha0)
								- Gamma.logGamma((a_sum[m]-1)*alpha1 + K*alpha0);
						log_diff -= Gamma.logGamma(ndsum[m] + a_sum[m]*alpha1 + K*alpha0)
								- Gamma.logGamma(ndsum[m] + (a_sum[m]-1)*alpha1 + K*alpha0);

						ratio = Math.exp(log_diff) * pi_a[m] / (1.0-pi_a[m]);
						p = ratio / (1.0 + ratio);
						if (rand.nextDouble() > p) { 
							a[m][k] = false;
							a_sum[m] --;
						}
					} else if (!a[m][k]) {
						log_diff = Gamma.logGamma((a_sum[m]+1)*alpha1 + K*alpha0)
								- Gamma.logGamma(a_sum[m]*alpha1 + K*alpha0);
						log_diff -= Gamma.logGamma(ndsum[m] + (a_sum[m]+1)*alpha1 + K*alpha0)
								- Gamma.logGamma(ndsum[m] + a_sum[m]*alpha1 + K*alpha0);

						ratio = Math.exp(log_diff) * pi_a[m] / (1.0-pi_a[m]);
						p = ratio / (1.0 + ratio);
						if (rand.nextDouble() < p) { 
							a[m][k] = true;
							a_sum[m] ++;
						}
					}
				}

				BetaDistribution betaDist = new BetaDistribution(rand, gamma1 + a_sum[m], gamma0 + K - a_sum[m]);
				pi_a[m] = betaDist.sample();
			}
		}
	}
	//����Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[documents.length][K];
		for (int d = 0; d < documents.length; d++) {
			for (int k = 0; k < K; k++) {
				int x = a[d][k] ? 1 : 0;
				theta[d][k] = (nd[d][k] + + x*alpha1 + alpha0) / (ndsum[d] + K * alpha0);
			}
		}
		return theta;
	}
	//����Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		return phi;
	}
	//����PhiBar
	public double[][] estimatePhiBar() {
		double[][] phi_bar = new double[K][E];
		for (int k = 0; k < K; k++) {
			for (int e = 0; e < E; e++) {
				phi_bar[k][e] = (ne[e][k] + beta_bar) / (nesum[k] + E * beta_bar);
			}
		}
		return phi_bar;
	}
}
