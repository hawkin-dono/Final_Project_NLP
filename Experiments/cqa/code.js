import { TextGeneration } from "deepinfra";
import * as fs from "fs";
import { DEEPINFRA_API_KEY } from "./secret.js"

const questions = fs
  .readFileSync("./dev_rand_split.jsonl", { encoding: "utf-8" })
  .split("\n")
  .filter((x) => x !== "")
  .map((x) => JSON.parse(x));

const prompt = ({
  id,
  question_concept,
  stem,
  formatted_choices,
}) => `You are a highly intelligent assistant specializing in solving multiple-choice commonsense reasoning questions. Your task is to analyze the given context, assess the question and its answer choices, and select the most logical and contextually accurate answer.
### Guidelines for the Task:
**Format the Final Answer**: Present your answer in the format \`<answer>C</answer>\`, replacing "C" with the correct letter corresponding to the chosen option.
### Task Input:
**Question ID**: ${id}  
**Concept**: ${question_concept}  
**Stem**: ${stem}  
**Choices**:  
${formatted_choices}
### Task Output:
2. **Final Answer**: Format your final answer as \`<answer>C</answer>\` where "C" is the correct option letter.
-------------------------------------------------------------------------
AREA FOR THINKING PROCESS AND ANSWER
`;

function getPromptFromRow(row) {
  // Define the prompt format (ensure this is set correctly)
  const formattedPrompt = prompt({
    id: row.id,
    question_concept: row.question.question_concept,
    stem: row.question.stem,
    formatted_choices: row.question.choices
      .map((choice) => `${choice.label}. ${choice.text}`)
      .join("\n"),
  });

  return formattedPrompt;
}

function postProcess(response) {
  const pattern = /<answer>(.*?)<\/answer>/;
  const match = response.match(pattern);
  if (match) {
    return match[1];
  }
  return null;
}

async function generate(MODEL_URL, input, stop) {
  const client = new TextGeneration(MODEL_URL, DEEPINFRA_API_KEY);
  const res = await client.generate({
    input,
    stop,
  });

  return res.results[0].generated_text;
}

const llama = (x) =>
  generate(
    "https://api.deepinfra.com/v1/inference/meta-llama/Meta-Llama-3-8B-Instruct",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" +
      x +
      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ["<|eot_id|>"]
  );

const gwen = (x) =>
  generate(
    "https://api.deepinfra.com/v1/inference/Qwen/Qwen2.5-72B-Instruct",
    "<|im_start|>user\n" + x + "<|im_end|>\n<|im_start|>assistant\n",
    ["<|im_start|>", "<|im_end|>", "</s>", "<|endoftext|>"]
  );

const gemma = (x) =>
  generate(
    "https://api.deepinfra.com/v1/inference/google/gemma-2-9b-it",
    "<bos><start_of_turn>user\n" + x + "<end_of_turn>\n<start_of_turn>model\n",
    ["<eos>", "<end_of_turn>"]
  );

// Define AI models to test
const AI_MODELS = [
  { name: 'gemma', func: gemma },
  { name: 'llama', func: llama },
  { name: 'gwen', func: gwen }
];

// Initialize tracking maps
const results = {
  correctAnswer: new Map(AI_MODELS.map(ai => [ai.name, 0])),
  cantExtract: new Map(AI_MODELS.map(ai => [ai.name, 0])),
  failCounter: new Map(AI_MODELS.map(ai => [ai.name, 0]))
};

const getAnswerFromRow = async (row, aiFunction, aiName) => {
  console.log(aiName + " is processing " + row.id);
  const generated = await aiFunction(getPromptFromRow(row));
  console.log(
    aiName + " generated this: " + JSON.stringify(generated) + " from " + row.id
  );
  const processedAnswer = postProcess(generated);
  console.log(
    aiName + " processed answer: ",
    JSON.stringify(processedAnswer) + " from " + row.id
  );
  if (processedAnswer === null) {
    console.log("Can't extract answer from " + aiName + " for " + row.id);
    results.cantExtract.set(aiName, results.cantExtract.get(aiName) + 1);
  } else if (processedAnswer.toLowerCase() === row.answerKey.toLowerCase()) {
    console.log("Correct answer from " + aiName + " for " + row.id);
    results.correctAnswer.set(aiName, results.correctAnswer.get(aiName) + 1);
  } else {
    console.log("Incorrect answer from " + aiName + " for " + row.id);
  }
};

// Sleep function for retry delays
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Process a single row with retry logic
async function processRowWithRetry(row, aiFunc, aiName, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await getAnswerFromRow(row, aiFunc, aiName);
      return true;
    } catch (error) {
      console.log(`Attempt ${attempt} failed for ${aiName} on row ${row.id}:`, error);
      if (attempt === maxRetries) {
        console.log(`All ${maxRetries} attempts failed for ${aiName} on row ${row.id}`);
        results.failCounter.set(aiName, results.failCounter.get(aiName) + 1);
        return false;
      }
      await sleep(5000); // Wait 5 seconds before retry
    }
  }
}

// Process rows in parallel with concurrency control
async function processDatasetInParallel(concurrencyLimit = 100) {
  const queue = [];
  
  // Create all jobs
  for (const question of questions) {
    for (const ai of AI_MODELS) {
      queue.push({ question, ai });
    }
  }

  // Process jobs with concurrency limit
  const activeJobs = new Set();
  const processNextJob = async () => {
    if (queue.length === 0) return;
    
    const { question, ai } = queue.shift();
    const jobPromise = processRowWithRetry(question, ai.func, ai.name)
      .finally(() => {
        activeJobs.delete(jobPromise);
        if (queue.length > 0) {
          processNextJob();
        }
      });
    
    activeJobs.add(jobPromise);
  };

  // Start initial batch of jobs
  for (let i = 0; i < Math.min(concurrencyLimit, queue.length); i++) {
    processNextJob();
  }

  // Wait for all jobs to complete
  while (activeJobs.size > 0) {
    await Promise.race([...activeJobs]);
  }
}

// Main execution
async function main() {
  console.log('Starting parallel processing...');
  const startTime = Date.now();

  await processDatasetInParallel(100);

  const endTime = Date.now();
  const duration = (endTime - startTime) / 1000;

  console.log('\nProcessing complete!');
  console.log(`Total time: ${duration} seconds`);
  console.log('\nResults:');
  console.log('Correct answers:', Object.fromEntries(results.correctAnswer));
  console.log('Unextractable answers:', Object.fromEntries(results.cantExtract));
  console.log('Failed jobs:', Object.fromEntries(results.failCounter));
}

main().catch(console.log);