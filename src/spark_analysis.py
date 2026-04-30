from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Healthcare Analysis").getOrCreate()

df = spark.read.csv("data/processed/healthcare_processed.csv", header=True, inferSchema=True)

df.createOrReplaceTempView("patients")

print("\nReadmission Rate by Age Group:")
spark.sql("""
SELECT age_group,
       COUNT(*) as total,
       SUM(readmitted_30_days) as readmissions,
       ROUND(SUM(readmitted_30_days)/COUNT(*), 2) as rate
FROM patients
GROUP BY age_group
""").show()

print("\nHigh Risk Distribution:")
spark.sql("""
SELECT high_risk_flag,
       COUNT(*) as count
FROM patients
GROUP BY high_risk_flag
""").show()

print("\nInsurance Risk Analysis:")
spark.sql("""
SELECT insurance_type,
       COUNT(*) as total,
       SUM(readmitted_30_days) as readmissions
FROM patients
GROUP BY insurance_type
ORDER BY readmissions DESC
""").show()