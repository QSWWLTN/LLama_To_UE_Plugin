#include "llamaSubsystem.h"

#include "common.h"

void UllamaSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
	Super::Initialize(Collection);

}

void UllamaSubsystem::Deinitialize() {
	Super::Deinitialize();

	Close();
	bIsInitFinish = false;

	llama_backend_free();
}

void UllamaSubsystem::Init() {
	llama_backend_init();

	bIsInitFinish = true;
}

void UllamaSubsystem::OpenModel(FString ModelPath, const FllamaModelParam& InModelParam) {

	if (IsLoadModel() || IsDecodeToken()) {
		return;
	}
	Close();

	OnLoadModelFinish.AddWeakLambda(this, [=](bool IsSuccess) {

		bIsLoadModel = false;

		if (!IsSuccess) {
			Close();
			OnInitModelFinish.Broadcast(false);
			return;
		}

		FString Prompt;
		if (FFileHelper::LoadFileToString(Prompt, *FPaths::ConvertRelativePathToFull(InModelParam.DefaultPromptPath))) {
			SendMassageToSystem(TEXT("system"), Prompt, false);
		}

		OnInitModelFinish.Broadcast(true);
	});

	bIsLoadModel = true;

	AsyncTask(ENamedThreads::HighThreadPriority, [=]() {
		ModelParam = llama_model_default_params();
		ModelParam.use_mmap = InModelParam.bUsemmap;
		ModelParam.use_mlock = InModelParam.bUseMlock;
		ModelParam.n_gpu_layers = InModelParam.UseGPULayer < 0.f ? 99 : InModelParam.UseGPULayer;

		LoadModel = llama_model_load_from_file(TCHAR_TO_UTF8(*ModelPath), ModelParam);
		if (!LoadModel) {
			UE_LOG(LogTemp, Error, TEXT("Try LoadMode: %s An Error Occurred, Please Check ModelPath or ModelType"), *ModelPath);
			AsyncTask(ENamedThreads::GameThread, [=]() {
				OnLoadModelFinish.Broadcast(false);
			});
			return;
		}

		//这里到时候有空再加
		ContextParam = llama_context_default_params();
		ContextParam.n_ctx = 4096;
		ContextParam.n_batch = 4096;

		ModelContext = llama_init_from_model(LoadModel, ContextParam);
		if (!ModelContext) {
			UE_LOG(LogTemp, Error, TEXT("Try Create Context Error"));
			AsyncTask(ENamedThreads::GameThread, [=]() {
				OnLoadModelFinish.Broadcast(false);
			});
			return;
		}

		const llama_vocab* pVocab = llama_model_get_vocab(LoadModel);

		if (!FFileHelper::LoadFileToString(ModelTemplate, *InModelParam.DefaultTemplatePath)) {
			ModelTemplate = UTF8_TO_TCHAR(llama_model_chat_template(LoadModel, nullptr));
		}

		llama_sampler_chain_params SamplerParam = llama_sampler_chain_default_params();
		ModelSampler = llama_sampler_chain_init(SamplerParam);

		llama_sampler_chain_add(ModelSampler, llama_sampler_init_min_p(InModelParam.MinP, 1));
		llama_sampler_chain_add(ModelSampler, llama_sampler_init_temp(InModelParam.Temperature));
		llama_sampler_chain_add(ModelSampler, llama_sampler_init_dist(InModelParam.bUseFixedSeed ? InModelParam.Seed : LLAMA_DEFAULT_SEED));

		AsyncTask(ENamedThreads::GameThread, [=]() {
			OnLoadModelFinish.Broadcast(true);
		});
	});
}

void UllamaSubsystem::Close() {

	if (ModelContext) {
		llama_kv_cache_clear(ModelContext);
		llama_free(ModelContext);
		ModelContext = nullptr;
	}

	if (ModelSampler) {
		llama_sampler_free(ModelSampler);
		ModelSampler = nullptr;
	}

	if (LoadModel) {
		llama_model_free(LoadModel);
		LoadModel = nullptr;
	}

	for (auto& Data : MassageList) {
		free(const_cast<char*>(Data.content));
	}

	ModelBatch = llama_batch();
	ModelTemplate = FString();

	TokenBuff.Empty();
	ChatTemplateBuff.Empty();
	MassageList.Empty();
}

void UllamaSubsystem::Stop() {
	
}

void UllamaSubsystem::SendMassgae(const FString& InMassage) {

	if (!IsRunModel() || IsDecodeToken()) {
		OnSendMassageError.Broadcast();
		return;
	}
	SendMassageToSystem(TEXT("user"), InMassage);
}

void UllamaSubsystem::SendMassageToSystem(const FString& Role, const FString& Context, bool GetMassage) {

	if (IsDecodeToken()) {
		OnSendMassageError.Broadcast();
		return;
	}

	if (Context.IsEmpty()) {
		return;
	}

	AddMassageToList(Role, Context);

	int Size = llama_chat_apply_template(TCHAR_TO_UTF8(*ModelTemplate), MassageList.GetData(), MassageList.Num(), true, ChatTemplateBuff.GetData(), ChatTemplateBuff.Num());
	if (Size > ChatTemplateBuff.Num()) {

		ChatTemplateBuff.Init(0, Size);
		Size = llama_chat_apply_template(TCHAR_TO_UTF8(*ModelTemplate), MassageList.GetData(), MassageList.Num(), true, ChatTemplateBuff.GetData(), ChatTemplateBuff.Num());
	}
	if (Size < 0) {
		UE_LOG(LogTemp, Warning, TEXT("Failed to apply the chat template\n"));
		OnSendMassageError.Broadcast();
		return;
	}

	std::string OutMassage(ChatTemplateBuff.GetData(), ChatTemplateBuff.Num());
	DecodeToken(OutMassage, GetMassage);
}

void UllamaSubsystem::DecodeToken(std::string InMassage, bool GetMassage) {
	if (!IsRunModel() || IsDecodeToken() || IsLoadModel()) {
		return;
	}

	const bool IsFirst = llama_get_kv_cache_used_cells(ModelContext) == 0;
	const llama_vocab* pVocab = llama_model_get_vocab(LoadModel);
	const int32 MassageTokenSize = -llama_tokenize(pVocab, InMassage.c_str(), InMassage.size(), NULL, 0, IsFirst, true);

	MassageTokenBuff.Init(0, MassageTokenSize);
	if (llama_tokenize(pVocab, InMassage.c_str(), InMassage.size(), MassageTokenBuff.GetData(), MassageTokenBuff.Num(), IsFirst, true) < 0) {
		UE_LOG(LogTemp, Error, TEXT("Failed to Tokenize the Prompt"));
		return;
	}
	if (!GetMassage) {
		return;
	}

	ModelBatch = llama_batch_get_one(MassageTokenBuff.GetData(), MassageTokenBuff.Num());
	bIsDecodeToken = true;

	AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [=]() {
		FString RetMassage;
		while (true)
		{
			int32 Size = llama_n_ctx(ModelContext);
			int32 CtxUseSize = llama_get_kv_cache_used_cells(ModelContext);

			if (CtxUseSize + ModelBatch.n_tokens > Size) {
				UE_LOG(LogTemp, Warning, TEXT("Context Data to loog"));
				bIsDecodeToken = false;
				return;
			}

			uint8 DecodeType = llama_decode(ModelContext, ModelBatch);
			if (DecodeType) {
				UE_LOG(LogTemp, Warning, TEXT("Failed to Decode, Type: %d"), DecodeType);
				bIsDecodeToken = false;
				return;
			}

			CurrToken = llama_sampler_sample(ModelSampler, ModelContext, -1);
			if (llama_vocab_is_eog(pVocab, CurrToken)) {
				break;
			};

			char Buff[256];
			int BuffSize = llama_token_to_piece(pVocab, CurrToken, Buff, sizeof(Buff), 0, false);
			if (BuffSize < 0) {
				GGML_ABORT("failed to convert token to piece\n");
			}
			std::string Str(Buff, BuffSize);
			ModelBatch = llama_batch_get_one(&CurrToken, 1);

			FString Temp = UTF8_TO_TCHAR(Str.c_str());
			if (Temp.Contains("<think>")) {
				bIsThinking = true;
			}
			else if (Temp.Contains("</think>")) {
				bIsThinking = false;
				if (!bIsShowThink) {
					Temp = TEXT("");
				}
			}
			if (bIsThinking) {
				AsyncTask(ENamedThreads::GameThread, [=]() {
					OnReadThinkMassage.Broadcast(Temp);
				});
			}

			if (!bIsShowThink && bIsThinking) {
				continue;
			}
			RetMassage.Append(Temp);

			AsyncTask(ENamedThreads::GameThread, [=]() {
				OnReadMassage.Broadcast(RetMassage);
			});
		}

		AsyncTask(ENamedThreads::GameThread, [=]() {
			OnReadAllMassage.Broadcast(RetMassage);
			AddMassageToList(TEXT("assistant"), RetMassage);

			int32 Size = llama_chat_apply_template(TCHAR_TO_UTF8(*ModelTemplate), MassageList.GetData(), MassageList.Num(), false, nullptr, 0);
			if (Size < 0) {
				UE_LOG(LogTemp, Error, TEXT("Can't Use ChatTemplate"));
			}
			bIsDecodeToken = false;
		});
	});
}