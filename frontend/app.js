const form = document.getElementById("emotion-form");
const textInput = document.getElementById("text-input");
const apiUrlInput = document.getElementById("api-url");
const statusEl = document.getElementById("status");
const btnExample = document.getElementById("btn-example");

const resultSection = document.getElementById("result");
const finalEmotionEl = document.getElementById("final-emotion");
const finalExplanationEl = document.getElementById("final-explanation");
const modelTableBody = document.querySelector("#model-table tbody");
const emotionDistributionEl = document.getElementById("emotion-distribution");

// Example texts to cycle through
const EXAMPLES = [
  "I am so happy and excited today!",
  "This makes me so angry and frustrated.",
  "I feel really sad and alone right now.",
  "I love spending time with my family.",
  "I am so scared about tomorrow.",
  "Wow, I didn't expect that at all!"
];
let exampleIndex = 0;

btnExample.addEventListener("click", () => {
  textInput.value = EXAMPLES[exampleIndex];
  exampleIndex = (exampleIndex + 1) % EXAMPLES.length;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const text = textInput.value.trim();
  const apiUrl = (apiUrlInput.value || "").trim();

  if (!text) {
    statusEl.textContent = "Please enter some text first.";
    statusEl.className = "status error";
    return;
  }

  if (!apiUrl) {
    statusEl.textContent =
      "Please provide the URL of your Python prediction API (e.g. https://your-backend.onrender.com/predict).";
    statusEl.className = "status error";
    return;
  }

  statusEl.textContent = "Sending text to the model API...";
  statusEl.className = "status";
  resultSection.classList.add("hidden");

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const bodyText = await response.text();
      throw new Error(
        `API returned ${response.status}. Response body: ${bodyText.slice(
          0,
          300
        )}...`
      );
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    renderResult(data);
    statusEl.textContent = "Prediction received successfully.";
    statusEl.className = "status success";
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Failed to get prediction: ${err.message}`;
    statusEl.className = "status error";
  }
});

/**
 * Expected JSON format from API:
 * {
 *   "final_emotion": "joy",
 *   "per_model": {
 *      "SVM": "joy",
 *      "Logistic Regression": "joy",
 *      "Random Forest": "joy",
 *      "XGBoost": "joy",
 *      "Naive Bayes": "joy",
 *      "Decision Tree": "joy"
 *   }
 * }
 */
function renderResult(payload) {
  const { final_emotion, per_model } = payload;
  if (!per_model || typeof per_model !== "object") {
    throw new Error("Unexpected API response shape.");
  }

  // Fill per-model table
  modelTableBody.innerHTML = "";
  Object.entries(per_model).forEach(([modelName, emotion]) => {
    const tr = document.createElement("tr");
    const tdModel = document.createElement("td");
    const tdEmotion = document.createElement("td");

    tdModel.textContent = modelName;
    tdEmotion.textContent = emotion;

    tr.appendChild(tdModel);
    tr.appendChild(tdEmotion);
    modelTableBody.appendChild(tr);
  });

  // Compute distribution
  const counts = {};
  Object.values(per_model).forEach((emotion) => {
    if (!counts[emotion]) counts[emotion] = 0;
    counts[emotion] += 1;
  });

  emotionDistributionEl.innerHTML = "";
  Object.entries(counts).forEach(([emotion, count]) => {
    const li = document.createElement("li");
    li.className = "chip";
    li.innerHTML = `<span class="label">${emotion}</span><span class="count">${count} model(s)</span>`;
    emotionDistributionEl.appendChild(li);
  });

  // Final emotion and explanation
  finalEmotionEl.textContent = final_emotion || inferMajority(counts);

  const majorityEmotion = finalEmotionEl.textContent;
  const totalModels = Object.keys(per_model).length;
  const majorityCount = counts[majorityEmotion] || 0;

  finalExplanationEl.textContent = `“${majorityEmotion}” was selected as the final emotion because ${majorityCount} out of ${totalModels} models agreed on this label.`;

  resultSection.classList.remove("hidden");
}

function inferMajority(counts) {
  let bestEmotion = "";
  let bestCount = -1;
  Object.entries(counts).forEach(([emotion, count]) => {
    if (count > bestCount) {
      bestCount = count;
      bestEmotion = emotion;
    }
  });
  return bestEmotion;
}


