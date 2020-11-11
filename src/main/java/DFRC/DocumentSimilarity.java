package DFRC;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class DocumentSimilarity {
    private static double getCosineSimilarity(Vector v1, Vector v2) {
        return v1.dot(v2) / (Vectors.norm(v1, 2) * (Vectors.norm(v2, 2)));
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
            .appName("")
            .config("spark.master", "local")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        String sourceContent = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.";
        // 특수문자를 없애되 어퍼스트로피로 쓸 따옴포와 split의 기준이 될 \n, 공백은 제외함 
        String sourceReplaced = sourceContent.replaceAll("[^a-zA-Z0-9| |\n|']", "");
        List<String> sourceSplitted = Arrays.asList(sourceReplaced.split("\n| "));

        Row sourceRow = RowFactory.create(sourceSplitted);

        String destContent1 = "Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, \"Lorem ipsum dolor sit amet..\", comes from a line in section 1.10.32.";
        String destContent2 = "The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested Sections 1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H Rackham.";
        String destContent3 = "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).";
        String destContent4 = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and less recently with laptop publishing software like Alps PageMarker including versions of Lorem Ipsum.";

        List<String> destContents = Arrays.asList(destContent1, destContent2, destContent3, destContent4);
        StructType schema = new StructType(new StructField[] {
            new StructField("words", new ArrayType(DataTypes.StringType, true), false, Metadata.empty()) });

        List<Double> results = new ArrayList<Double>(destContents.size());
        
        for (String destContent : destContents) {
            List<Row> rowsForMatrix = new ArrayList<Row>();
            rowsForMatrix.add(sourceRow);
            String destReplaced = destContent.replaceAll("[^a-zA-Z0-9| |\n|']", "");
            List<String> destSplitted = Arrays.asList(destReplaced.split("\n| "));
            
            rowsForMatrix.add(RowFactory.create(destSplitted));
            Dataset<Row> dataSet = spark.createDataFrame(rowsForMatrix, schema);
            CountVectorizerModel CVM = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10000)
                .setMinDF(1)
                .fit(dataSet);
            Dataset<Row> cvmResult = CVM.transform(dataSet);
            
            JavaRDD<DenseVector> denseVectors = cvmResult
                .select("features")
                .javaRDD()
                .map((row) -> Vectors.parse(row.get(0).toString()).toDense());
            // dest 문서 하나씩 비교
            results.add(getCosineSimilarity(denseVectors.first(), denseVectors.collect().get(1)));
        }        
        
        for(double result : results){
            System.out.println(result);
        }
        spark.stop();
    }
}
