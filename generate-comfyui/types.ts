export type ModelTags = {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    parent_model: string;
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
};

export type GetModelsResponse = {
  models: ModelTags[];
};

export type Message = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  images?: string[];
};
