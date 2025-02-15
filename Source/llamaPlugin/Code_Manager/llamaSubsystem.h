#pragma once

#include "CoreMinimal.h"
#include "Subsystems/EngineSubsystem.h"

#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-vocab.h"

#include "llamaSubsystem.generated.h"

USTRUCT(BlueprintType)
struct FllamaModelParam {
	GENERATED_USTRUCT_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "加载到GPU的层数，小于0的时候将使用最大可用的层数"))
		int32 UseGPULayer = -1;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "最小P采样"))
		float MinP = 0.05f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "温度"))
		float Temperature = 0.6f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "使用固定种子"))
		bool bUseFixedSeed = false;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "固定种子", EditCondition = "bUseFixedSeed"))
		int32 Seed = 1;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "最大token数量"))
		int32 TokenBuffMaxSize = 4096;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "提示词文件地址"))
		FString DefaultPromptPath;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "模板文件地址，如果没有指定将使用默认模板"))
		FString DefaultTemplatePath;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "尝试使用mmap"))
		bool bUsemmap = false;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ToolTip = "强制系统将模型保留在RAM中"))
		bool bUseMlock = false;
};

DECLARE_MULTICAST_DELEGATE_OneParam(FOnLoadModelFinish, bool IsSuccess);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnInitModelFinish, bool, IsSuccess);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnReadThinkMassage, FString, ThinkMassage);

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnReadMassage, FString, OutMassage);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnReadAllMassage, FString, AllMassage);

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnSendMassageError);

UCLASS()
class LLAMAPLUGIN_API UllamaSubsystem : public UEngineSubsystem
{
	GENERATED_BODY()
public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

public:
	UFUNCTION(BlueprintCallable)
		void Init();
	UFUNCTION(BlueprintCallable)
		void OpenModel(FString ModelPath, const FllamaModelParam& InModelParam);
	UFUNCTION(BlueprintCallable)
		void Close();
	UFUNCTION(BlueprintCallable)
		void Stop();

	UFUNCTION(BlueprintCallable)
		void SendMassgae(const FString& InMassage);

	UFUNCTION(BlueprintPure)
		bool IsRunModel() const { return LoadModel && ModelContext; };
	UFUNCTION(BlueprintPure)
		bool IsInitFinish() const { return bIsInitFinish; };
	UFUNCTION(BlueprintPure)
		bool IsDecodeToken() const { return bIsDecodeToken; };
	UFUNCTION(BlueprintPure)
		bool IsLoadModel() const { return bIsLoadModel; };

protected:
	void SendMassageToSystem(const FString& Role, const FString& Context, bool GetMassage = true);
	void DecodeToken(std::string InMassage, bool GetMassage = true);

	void AddMassageToList(const FString& Role, const FString& Context) { MassageList.Add({ strdup(TCHAR_TO_UTF8(*Role)), strdup(TCHAR_TO_UTF8(*Context)) }); };

public:
	FOnLoadModelFinish OnLoadModelFinish;

public:
	UPROPERTY(BlueprintAssignable)
		FOnInitModelFinish OnInitModelFinish;

	UPROPERTY(BlueprintAssignable)
		FOnReadThinkMassage OnReadThinkMassage;

	UPROPERTY(BlueprintAssignable)
		FOnReadMassage OnReadMassage;
	UPROPERTY(BlueprintAssignable)
		FOnReadAllMassage OnReadAllMassage;

	UPROPERTY(BlueprintAssignable)
		FOnSendMassageError OnSendMassageError;

public:
	UPROPERTY(BlueprintReadWrite)
		bool bIsShowThink = false;

protected:
	llama_model* LoadModel;
	llama_context* ModelContext;
	llama_sampler* ModelSampler;

	FString ModelTemplate;

	llama_batch ModelBatch;
	llama_token CurrToken;

	llama_model_params ModelParam;
	llama_context_params ContextParam;

	TArray<llama_token> TokenBuff;
	TArray<llama_token> MassageTokenBuff;
	TArray<char> ChatTemplateBuff;
	TArray<llama_chat_message> MassageList;

protected:
	bool bIsInitFinish = false;
	bool bIsLoadModel = false;
	bool bIsDecodeToken = false;
	bool bIsThinking = false;
};
