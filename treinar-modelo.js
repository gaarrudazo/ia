// treinar-modelo.js
const tf = require('@tensorflow/tfjs-node');

// Defina a função log para usar console.log (para ambiente Node.js)
function log(message) {
  console.log(message);
}

// Função para treinar o modelo
async function treinarModelo() {
  // Exemplo: Gera dados aleatórios para treinamento
  const numAmostras = 100;
  const numFeatures = 11;
  
  // Cria um tensor de entrada com distribuição normal (substitua por seus dados reais)
  const X = tf.randomNormal([numAmostras, numFeatures]);
  // Cria um tensor de saída (target) com valores entre 0 e 1
  const y = tf.randomUniform([numAmostras, 1], 0, 1);

  // Define um modelo sequencial simples
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numFeatures],
    units: 32,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid' // Saída entre 0 e 1
  }));

  // Compila o modelo
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  log("Iniciando treinamento...");
  // Treina o modelo
  await model.fit(X, y, {
    epochs: 50,
    batchSize: 16,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, mae = ${logs.mae.toFixed(4)}`);
      }
    }
  });
  log("Treinamento concluído!");

  // Define o caminho para salvar o modelo no formato TensorFlow.js
  // O prefixo "file://" é necessário para salvar localmente
  const savePath = 'file://' + __dirname + '/modelo_tfjs';
  await model.save(savePath);
  log(`Modelo salvo em: ${savePath}`);
}

// Executa o treinamento
treinarModelo()
  .then(() => {
    log("Processo finalizado!");
  })
  .catch(err => {
    console.error("Erro durante o treinamento:", err);
  });
