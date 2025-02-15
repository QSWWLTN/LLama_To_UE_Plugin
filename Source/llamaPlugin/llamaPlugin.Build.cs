// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class llamaPlugin : ModuleRules
{
	
	private string ModulePath
    {
        get { return ModuleDirectory; }
    }

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModulePath, "../../ThirdParty/")); }
    }
	
	public llamaPlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		PublicIncludePaths.AddRange(
			new string[] {
				Path.Combine(ThirdPartyPath, "include"),
				Path.Combine(ThirdPartyPath, "include", "Cuda")
			}
		);
		
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
		);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...
			}
		);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
		);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
		);
		
		if ((Target.Platform == UnrealTargetPlatform.Win64))
		{
			string[] DllList = {
                "ggml",
                "ggml-base",
                "ggml-cpu",
                "ggml-cuda",
                "llama"
            };

			foreach (var DllName in DllList) {
                PublicAdditionalLibraries.Add(Path.Combine(ThirdPartyPath, "lib", DllName + ".lib"));
            }
			
			var CudaLib = Directory.GetFiles(Path.Combine(ThirdPartyPath, "lib", "Cuda"), "*.lib");
			foreach(var CudaName in CudaLib){
				PublicAdditionalLibraries.Add(Path.Combine(ThirdPartyPath, "lib", "Cuda", CudaName));
			}
			
			PublicDelayLoadDLLs.Add("cudart64_12.dll");
			RuntimeDependencies.Add(Path.Combine(ThirdPartyPath, "dll", "cudart64_12.dll"));
		}
	}
}
