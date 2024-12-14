import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import {
	ApiHandlerOptions,
	azureOpenAiDefaultApiVersion,
	ModelInfo,
	openAiModelInfoSaneDefaults,
} from "../../shared/api"
import { ApiHandler } from "../index"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"

export class OpenAiHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		// Azure API shape slightly differs from the core API shape: https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
		if (this.options.openAiBaseUrl?.toLowerCase().includes("azure.com")) {
			this.client = new AzureOpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: this.options.openAiApiKey,
				apiVersion: this.options.azureApiVersion || azureOpenAiDefaultApiVersion,
			})
		} else {
			this.client = new OpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: this.options.openAiApiKey,
			})
		}
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.options.openAiModelId ?? ""
		
		// Handle o1 family models which don't support streaming, non-1 temp, or system prompt
		if (modelId === "o1-preview" || modelId === "o1-mini" || modelId === "o1" ||  modelId === "o1-pro") {
			const response = await this.client.chat.completions.create({
				model: modelId,
				messages: [{ role: "user", content: systemPrompt }, ...convertToOpenAiMessages(messages)],
			})
			yield {
				type: "text",
				text: response.choices[0]?.message.content || "",
			}
			yield {
				type: "usage",
				inputTokens: response.usage?.prompt_tokens || 0,
				outputTokens: response.usage?.completion_tokens || 0,
			}
			return
		}

		// Handle other models with streaming support
		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]
		const stream = await this.client.chat.completions.create({
			model: modelId,
			messages: openAiMessages,
			temperature: 0,
			stream: true,
			stream_options: { include_usage: true },
		})
		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}
			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.openAiModelId ?? "",
			info: openAiModelInfoSaneDefaults,
		}
	}
}
