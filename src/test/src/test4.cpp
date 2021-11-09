#include<fstream>
#include<mlpack/methods/kmeans/kmeans.hpp>
#include<armadillo>

using namespace std;

int main()
{
	int k = 2;
	int dim = 5;
	int samples = 5;
	int max_iter = 100;

	arma::mat data(dim, samples, arma::fill::zeros);

	int  i = 0;

	for(; i < samples/2; i++)
		//arma::randn<arma::vec>(dim) == random noise vector of 2 dimentional
		data.col(i) = arma::vec({1, 1,1,1,1}) + 0.25 * arma::randn<arma::vec>(dim);

	for(; i < samples; i++)
		data.col(i) = arma::vec({2, 3,1,1,1}) + 0.25 * arma::randn<arma::vec>(dim);

	//arma row type to hold the clusters
	//arma mat to hold centroids

	arma::Row<size_t>clusters;
	arma::mat centroids;

	mlpack::kmeans::KMeans<> mlkmeans(max_iter);

	mlkmeans.Cluster(data, k, clusters, centroids);

	centroids.print("Centroids");

return 0;
}
