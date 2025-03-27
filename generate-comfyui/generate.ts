/**
 * Description:
 *
 * 1. Reads all .txt prompts from input folder and generate images using ComfyUI and provided workflow json.
 * 2. Copies all generated images to input folder.
 * 3. Summarizes all generations in output file saved to input folder.
 *
 * Usage:
 *
 * ```
 * deno run --allow-net --allow-read --allow-write generate.ts input-folder
 * ```
 *
 * where input-folder is the relative path to the input folder containing all the .txt prompts and where all generated images will be copied.
 *
 * Configurations:
 *
 * - WORKFLOW_PATH: relative path to Comfy workflow
 * - COMFY_OUTPUT_DIR: ComfyUI output folder where images are generated
 * - API_URL: ComfyUI API (API needs to be running before launching the script)
 */

import { join } from "@std/path/join";
import { basename } from "@std/path/basename";
import { extname } from "@std/path/extname";

const WORKFLOW_PATH = "";
const COMFY_OUTPUT_DIR = "";
const API_URL = "http://127.0.0.1:8188/prompt";

type ProcessEntry = {
  name: string;
  inputPath: string;
  outputPath?: string;
  prompt: string;
  done: boolean;
};

export type GenerateOptions = {
  apiURL?: string;
  comfyOutputDir?: string;
  worflowPath?: string;
};

export async function callComfy(
  apiURL: string,
  workflow: any,
  name: string,
  prompt: string,
) {
  const random_id = Math.random().toString(36).substring(2, 8);
  const fileName = `${name}-${random_id}`;
  workflow["34"].inputs.positive = prompt;
  workflow["45"].inputs.filename_prefix = fileName;

  const headers = new Headers();
  headers.append("Content-Type", "application/json");

  const response = await fetch(apiURL, {
    method: "POST",
    headers,
    body: JSON.stringify({ prompt: workflow }),
  });

  if (response.status !== 200) {
    console.error(response);
    throw new Error("Can't call Comfy API");
  }

  return fileName;
}

async function getQueue(inputDir: string) {
  const processQueue: ProcessEntry[] = [];
  for await (const entry of Deno.readDir(inputDir)) {
    if (entry.isFile && extname(entry.name) === ".txt") {
      const inputPath = join(inputDir, entry.name);
      const prompt = await Deno.readTextFile(inputPath);
      const data = {
        name: entry.name,
        inputPath,
        prompt,
        done: false,
      };
      processQueue.push(data);
    }
  }
  return processQueue;
}

export async function generate(inputDir: string, options?: GenerateOptions) {
  // process options
  const workflowPath = options?.worflowPath || WORKFLOW_PATH;
  const comfyOutputDir = options?.comfyOutputDir || COMFY_OUTPUT_DIR;
  const apiURL = options?.apiURL || API_URL;

  // setup queue
  const processQueue = await getQueue(inputDir);

  // setup workflow
  let workflow = undefined;
  try {
    const workflowFile = await Deno.readTextFile(workflowPath);
    workflow = JSON.parse(workflowFile);
  } catch (error) {
    console.error("❌ Failed to read or parse workflow file:", error);
  }

  // get next prompt from queue
  const getNext = () => {
    const waitings = processQueue.filter((entry) => !entry.done);
    if (waitings.length > 0) {
      return waitings[0];
    }
    return null;
  };

  // recursively call next in queue
  const callNext = async () => {
    const next = getNext();
    if (!next) {
      console.log("✅ All images were generated");

      // generate summary file

      const outputSummaryFile = `summary_${
        (new Date()).toISOString().split(".")[0].replace(":", "_")
      }.json`;

      Deno.writeFile(
        join(inputDir, outputSummaryFile),
        new TextEncoder().encode(JSON.stringify(processQueue)),
      );

      return;
    }

    const outputFilename = await callComfy(
      apiURL,
      workflow,
      next.name,
      next.prompt,
    );
    next.done = true;
    next.outputPath = join(comfyOutputDir, `${outputFilename}_00001_.png`);
    console.log("Processing: ", next);

    let n = 0;
    const MAX_INTERVALS = 20;
    const timerId = setInterval(async () => {
      let found = false;
      for await (const entry of Deno.readDir(comfyOutputDir)) {
        if (entry.isFile && basename(entry.name).startsWith(outputFilename)) {
          clearInterval(timerId);
          found = true;
          // copy generated image to input folder
          if (next.outputPath) {
            await new Promise((resolve) => setTimeout(resolve, 5000)); // making sure image is completed
            Deno.copyFile(
              next.outputPath,
              join(inputDir, basename(next.outputPath)),
            );
          }
          break;
        }
      }
      if (found) {
        await callNext();
      } else if (n < MAX_INTERVALS) {
        console.log(`⏳ Generating for ${(n + 1) * 5} sec...`);
        n++;
      } else {
        console.error(
          "❌ Something went wrong with image generation, check Comfy",
        );
        clearInterval(timerId);
      }
    }, 5000);
  };

  await callNext();
}

if (import.meta.main) {
  const inputDir = Deno.args[0];
  if (inputDir && (await Deno.stat(inputDir)).isDirectory) {
    await generate(inputDir);
  } else {
    console.error("❌ Missing input directory");
  }
}
