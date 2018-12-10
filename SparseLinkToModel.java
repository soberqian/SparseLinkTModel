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

	double alpha0 ;  //超参数 alpha0 = 1E-12

	double alpha1 ;  //超参数alpha1 = 0.1

	double beta;

	double gamma0;  //beta分布对应的参数

	double gamma1; //beta分布对应的参数

	double beta_bar;

	int[][] z;

	int[][] z_bar;

	int[][] nw;

	int[][] ne;

	int[][] nd;

	int[] nwsum;

	int[] nesum;

	int[] ndsum;

	double pi_a[];  //pi参数

	boolean a[][]; //文档主题选择器

	int a_sum[];  //文档主题的个数

	JDKRandomGenerator rand; //随机数生成器

	int iterations;

	public SparseLinkToModel(int[][] documents, int[][] entities, int V, int E) {

		this.documents = documents;
		this.entities = entities;
		this.V = V;
		this.E = E;
	}
	//初始化操作
	public void initialState() {
		rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		//贝塔分布生成
		BetaDistribution betaDist = new BetaDistribution(rand, gamma1 , gamma0);
		//文档总数
		int D = documents.length;
		//文档d中主题k生成的单词数目
		nd = new int[D][K];
		//文档m是否包含主题k
		a = new boolean[D][K]; 
		//文档m包含主题的数量
		a_sum = new int[D]; 
		//pi参数
		pi_a = new double[D];  
		//每篇文档包含的所有单词总数
		ndsum = new int[D];
		//主题k中单词v的数目
		nw = new int[V][K];
		//主题k对应的单词总数
		nwsum = new int[K];
		//主题k中对应的实体e的数目
		ne = new int[E][K];
		//主题k中实体的总数
		nesum = new int[K];
		//每篇文档单词对应的主题
		z = new int[D][];
		//每篇文档对应的实体对应的主题
		z_bar = new int[D][];
		//对每一个文档抽取pi，随机抽取值
		for (int d = 0; d < D; d++) {

			pi_a[d] = betaDist.sample();
			//刚开始初始化文档包含所有主题
			for (int k = 0; k < K; k++) {
				a[d][k] = true;
			}
			//文档包含的主题数目总和
			a_sum[d] = K;
		}
		//循环每一篇文档
		for (int d = 0; d < D; d++) {

			// words
			int Nd = documents[d].length;
			z[d] = new int[Nd];
			//循环每一个单词
			for (int n = 0; n < Nd; n++) {
				//随机赋值主题
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
				//更新统计
				updateCount(d, topic, documents[d][n], +1);
			}

			// 获取文档对应的所有实体个数
			int Ed = entities[d].length;
			//文档实体对应的主题
			z_bar[d] = new int[Ed];
			//循环文档的所有实体
			for (int m = 0; m < Ed; m++) {
				//随机赋值主题
				int topic = (int) (Math.random() * K);
				z_bar[d][m] = topic;
				//更新统计
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
		//初始化--分配主题
		initialState();
		//抽样更新
		for (int i = 0; i < this.iterations; i++) {

			System.out.println("iteration : " + i);
			gibbs();  //执行gibbs采样
			//稀疏度计算
			if(i % 1 == 0) {
				//抽取二元矩阵
				sampleBinaryAMatrix();
			}
		}
	}
	//gibbs采样
	public void gibbs() {
		//循环所有文档
		for (int d = 0; d < documents.length; d++) {
			//循环所有单词
			for (int n = 0; n < z[d].length; n++) {
				//
				int topic = sampleFullConditional(d, n);
				z[d][n] = topic;

			}
			// 循环所有实体
			for (int m = 0; m < z_bar[d].length; m++) {

				int topic = sampleFullConditionalEntity(d, m);
				z_bar[d][m] = topic;

			}
		}
	}
	//抽取单词的主题;输入参数为文档d以及文档d中的单词n
	int sampleFullConditional(int d, int n) {
		//获取原对应的主题
		int topic = z[d][n];
		//更新数目,减1
		updateCount(d, topic, documents[d][n], -1);
		//概率
		double[] p = new double[K];
		//循环每个主题
		for (int k = 0; k < K; k++) {
			int x = a[d][k] ? 1 : 0;
			p[k] = (nd[d][k] + x*alpha1 + alpha0) / (ndsum[d] + K * alpha0) * (nw[documents[d][n]][k] + beta)
					/ (nwsum[k] + V * beta);
		}
		//轮盘赌抽取新主题
		topic = sample(p);
		//更新统计
		updateCount(d, topic, documents[d][n], +1);
		//返回主题
		return topic;

	}
	//抽取实体的主题
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
	//轮盘赌
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

	//更新统计
	void updateCount(int d, int topic, int word, int flag) {
		//文档 d中的主题topic对应的单词数目加1
		nd[d][topic] += flag;
		//文档d
		ndsum[d] += flag;
		//主题topic对应的单词word数量加1
		nw[word][topic] += flag;
		//主题topic对应的单词总数加1
		nwsum[topic] += flag;
	}
	//更新统计
	void updateEntityCount(int d, int topic, int entity, int flag) {
		//主题topic中实体entity的数目增加1
		ne[entity][topic] += flag;
		//主题topic对应的总的实体数目加1
		nesum[topic] += flag;
		//文档 d中的主题topic对应的单词数目加1
		nd[d][topic] += flag;
		//文档对应的单词以及实体的总个数
		ndsum[d] += flag;
	}
	//抽取文档主题选择器
	public void sampleBinaryAMatrix() {
		int GIBBS_ITER = 1;
		//文档选择主题的个数
		a_sum = new int[documents.length];
		//循环每一篇文档
		for (int m = 0; m != documents.length; m++) {
			//循环每个主题
			for (int k = 0; k != K; k++) {
				//判断文档是否已经有该主题了,如果有则为true
				a[m][k] = (nd[m][k]) > 0;
				//文档m包含的主题个数+1
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
	//估计Theta
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
	//估计Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		return phi;
	}
	//估计PhiBar
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
