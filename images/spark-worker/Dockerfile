FROM apache/spark-py:v3.3.0
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    python3-pip \
    openjdk-11-jdk && \
    pip3 install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages including PyTorch CPU
COPY requirements.runtime.txt /tmp/requirements.runtime.txt
RUN pip3 install --no-cache-dir --default-timeout=300 -r /tmp/requirements.runtime.txt

# Download and setup Hadoop client for HDFS access
ENV HADOOP_VERSION=3.2.1
RUN curl -sL https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | tar -xz -C /opt/ && \
    mv /opt/hadoop-${HADOOP_VERSION} /opt/hadoop

# Set Hadoop environment
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Copy Hadoop config for HDFS
COPY images/spark-worker/core-site.xml /opt/hadoop/etc/hadoop/core-site.xml

RUN rm -rf /root/.cache/pip

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

WORKDIR /opt/spark
