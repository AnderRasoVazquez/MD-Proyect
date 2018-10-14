package kmeans;

import utils.Utils;
import weka.core.*;

import java.util.Arrays;
import java.util.Random;

public class KMeans {

    private Instance[] instances;
    private Instance[] centroids;
    private int[] centroidsInstances;
    private Instance[] prevCentroids;
    private boolean[][] belongingBits;
    private int k;
    private double tolerance;
    private DistanceFunction distance;
    private int maxIt;


    public static void main(String[] args) {
        Instances instances = Utils.loadInstances("/home/david/Documentos/Universidad/4º/Minería de Datos/Proyecto/files/verbal_autopsies_raw_tfidf.arff", 0);
        System.out.println(instances.numInstances());
        KMeans kmeans = new KMeans(instances, 96, "9-last");
        kmeans.formClusters(true);
    }

    public KMeans(Instances pInstances, int pClusters, String pAttrRange) {
        this(pInstances, pClusters, 0.1, pAttrRange);
    }

    public KMeans(Instances pInstances, int pClusters, double pTolerance, String pAttrRange) {
        this.instances = new Instance[pInstances.numInstances()];
        for (int i = 0; i < instances.length; i++) {
            this.instances[i] = pInstances.get(i);
        }
        this.k = Math.min(pClusters, this.instances.length);
        this.centroids = new Instance[this.k];
        this.centroidsInstances = new int[k];
        this.prevCentroids = new Instance[this.k];
        this.belongingBits = new boolean[this.instances.length][this.k];
        this.tolerance = pTolerance;
        Range attrRange = new Range();
        attrRange.setRanges(pAttrRange);
        attrRange.setUpper(pInstances.numAttributes() - 1);
        this.distance = new EuclideanDistance();
        this.distance.setInstances(pInstances);
        this.distance.setAttributeIndices(pAttrRange);
        this.maxIt = 50;
    }

    public void formClusters() {
        this.formClusters(false);
    }

    public String formClusters(boolean pVerbose) {
        // initialize centroids to random instances
        this.initializeCentroids();
        int it = 0;
        // iterate until centroids converge
        while (it < this.maxIt && !this.checkFinished()) {
            it++;
            if (pVerbose)
                System.out.println("It: " + it);
            // update belonging matrix
            this.updateBelongingBits();
            // obtain new centroids
            this.updateCentroids();
            if (pVerbose)
                System.out.println(String.format("Iteration nº %d finished", it));
        }
        // obtain final belonging matrix
        this.updateBelongingBits();
        String clusters = this.getClusters();
        if (pVerbose)
            System.out.print(clusters);
        return clusters;
    }

    private void initializeCentroids() {
        Random rng = new Random();
        for (int i = 0; i < this.centroids.length; i++) {
            Instance centroid;
            do {
                centroid = this.instances[rng.nextInt(this.instances.length)];

            } while (Arrays.asList(this.centroids).contains(centroid));
            centroids[i] = centroid;
        }
    }

    private void updateBelongingBits() {
        for (int c = 0; c < this.belongingBits.length; c++) {
            for (int r = 0; r < this.belongingBits[0].length; r++) {
                this.belongingBits[c][r] = false;
            }
        }
        for (int t = 0; t < this.instances.length; t++) {
            double min_diff = 99999;
            int min_i = -1;
            Instance instance = this.instances[t];
            for (int i = 0; i < this.centroids.length; i++) {
                Instance centroid = this.centroids[i];
                double diff = Arrays.equals(instance.toDoubleArray(), centroid.toDoubleArray()) ? 0 : distance.distance(instance, centroid);
                if (diff < min_diff) {
                    min_diff = diff;
                    min_i = i;
                } else if (diff == min_diff && this.centroidsInstances[i] < this.centroidsInstances[min_i]) {
                    min_diff = diff;
                    min_i = i;
                }
            }
            this.belongingBits[t][min_i] = true;
            this.centroidsInstances[min_i]++;
        }
    }

    private void updateCentroids() {
        // save previous centroids for furture comparison
        this.prevCentroids = this.centroids.clone();

        for (int i = 0; i < this.centroids.length; i++) {
            // nº instances that belong to centroid i
            int bits_i = 0;
            // sum of those instances
            Instance instance_sum = null;
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i]) {
                    bits_i++;
                    instance_sum = addInstances(instance_sum, this.instances[t]);
                }
            }
            this.centroids[i] = divideInstance(instance_sum, bits_i);
            this.centroidsInstances[i] = 0;
        }
    }

    private boolean checkFinished() {
        if (prevCentroids[0] == null) {
            return false;
        }
        for (int i = 0; i < this.centroids.length; i++) {
            Instance centroid = this.centroids[i];
            Instance prev = this.prevCentroids[i];
            if (distance.distance(centroid, prev) > this.tolerance) {
                return false;
            }
        }
        return true;
    }

    private static Instance addInstances(Instance pInstanceA, Instance pInstanceB) {
        if (pInstanceA == null)
            return pInstanceB;
        else if (pInstanceB == null)
            return pInstanceA;
        else {
            if (pInstanceA.numAttributes() != pInstanceB.numAttributes())
                return null;
            else {
                Instance res = pInstanceA.copy(pInstanceA.toDoubleArray());
                for (int i = 0; i < pInstanceA.numAttributes(); i++) {
                    // if attribute is numeric
                    if (pInstanceA.attribute(i).type() == Attribute.NUMERIC) {
                        res.setValue(i, pInstanceA.value(i) + pInstanceB.value(i));
                    }
                }
                return res;
            }
        }
    }

    private static Instance divideInstance(Instance pInstance, double pNum) {
        Instance res = pInstance.copy(pInstance.toDoubleArray());
        for (int i = 0; i < pInstance.numAttributes(); i++) {
            // if attribute is numeric
            if (pInstance.attribute(i).type() == Attribute.NUMERIC) {
                res.setValue(i, pInstance.value(i) / pNum);
            }
        }
        return res;
    }

    private String getClusters() {
        StringBuilder clusters = new StringBuilder();
        clusters.append(String.format("k: %d\n", this.k));
        int attrToPrint = 4;
        for (int i = 0; i < this.centroids.length; i++) {
            clusters.append(String.format("CLUSTER %d\n", i));
            clusters.append(String.format("\tcentroid: %s\n", instanceToString(this.centroids[i], attrToPrint)));
            clusters.append("\tinstances:\n");
            for (int t = 0; t < this.instances.length; t++) {
                if (this.belongingBits[t][i])
                    clusters.append(String.format("\t\t%s\n", instanceToString(this.instances[t], attrToPrint)));
            }
        }
        return clusters.toString();
    }

    private static String instanceToString(Instance pInstance, int pAttributes) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < pAttributes; i++) {
            Attribute attr = pInstance.attribute(i);
            if (attr.type() == Attribute.NUMERIC) {
                res.append(pInstance.value(i));
            } else {
                res.append(attr.value((int) pInstance.value(i)));
            }
            if (i < pAttributes -1) {
                res.append(", ");
            }
        }
        return res.toString();
    }

    private void saveCentroids(String pPath, boolean pPrevious) {
        Instances centroids = new Instances(this.centroids[0].dataset());
        centroids.delete();
        centroids.addAll(Arrays.asList(this.centroids));
        if (pPrevious) {
            centroids.addAll(Arrays.asList(this.prevCentroids));
        }
        Utils.saveInstances(centroids, pPath);
    }
}
