/* script.js */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const sidebar = document.getElementById("sidebar");
const toggleBtn = document.getElementById("toggleSidebar");
const statusText = document.getElementById("statusText");

const detectedLabel = document.getElementById("detectedLabel");
const confidenceText = document.getElementById("confidence");

const trainBtn = document.getElementById("trainBtn");
const startDetectBtn = document.getElementById("startDetect");
const saveModelBtn = document.getElementById("saveModel");
const modelFilesInput = document.getElementById("modelFiles");

const LABELS = ["HELLO", "YES", "NO", "THANK YOU", "HELP"];
const samples = {};
LABELS.forEach(l => samples[l] = []);

let model = null;
let detecting = false;
let latestLandmarks = null;

/* Sidebar toggle */
toggleBtn.addEventListener("click", () => {
  sidebar.classList.toggle("closed");
});

/* MediaPipe Hands */
const hands = new Hands({
  locateFile: file =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.6
});

hands.onResults(results => {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.multiHandLandmarks) return;

  const landmarks = results.multiHandLandmarks[0];
  latestLandmarks = landmarks;

  drawLandmarks(ctx, landmarks);
});

/* Camera */
const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({ image: video });
  },
  width: 1280,
  height: 720
});
camera.start();

video.addEventListener("loadeddata", () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
});

/* Train model */
trainBtn.addEventListener("click", async () => {
  const xs = [];
  const ys = [];

  LABELS.forEach((label, i) => {
    samples[label].forEach(sample => {
      xs.push(sample);
      ys.push(i);
    });
  });

  if (xs.length < 50) {
    statusText.innerText = "Not enough samples";
    return;
  }

  model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [63] }));
  model.add(tf.layers.dense({ units: LABELS.length, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  await model.fit(
    tf.tensor2d(xs),
    tf.oneHot(tf.tensor1d(ys, "int32"), LABELS.length),
    { epochs: 30 }
  );

  statusText.innerText = "Training complete";
});

/* Detect toggle */
startDetectBtn.addEventListener("click", () => {
  detecting = !detecting;
  startDetectBtn.innerText = detecting ? "Stop Detecting" : "Start Detecting";
});

/* Save model */
saveModelBtn.addEventListener("click", async () => {
  if (model) {
    await model.save("downloads://easetranslate-model");
  }
});

/* Load model */
modelFilesInput.addEventListener("change", async e => {
  model = await tf.loadLayersModel(tf.io.browserFiles(e.target.files));
  statusText.innerText = "Model loaded";
});

/* Draw landmarks */
function drawLandmarks(ctx, landmarks) {
  ctx.fillStyle = "#00ff00";
  landmarks.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}
