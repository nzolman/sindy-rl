grab_data:
	git clone https://huggingface.co/nzolman/sindy-rl_data ./data

unzip_data:
	tar -xvzf ./data/benchmarks/swingup.tar.gz -C ./data/benchmarks/
	tar -xvzf ./data/benchmarks/swimmer.tar.gz -C ./data/benchmarks/
	tar -xvzf ./data/benchmarks/cylinder.tar.gz -C ./data/benchmarks/

	tar -xvzf ./data/agents/swingup.tar.gz -C ./data/agents/
	tar -xvzf ./data/agents/swimmer.tar.gz -C ./data/agents/
	tar -xvzf ./data/agents/cylinder.tar.gz -C ./data/agents/

	tar -xvzf ./data/hydrogym/cylinder.tar.gz -C ./data/hydrogym/

# THIS COULD BE EXTREMELY DANGEROUS! 
# BE VERY CERTAIN YOU HAVE NOTHING NEW INSIDE THESE FOLDERS WORTH SAVING
remove_data_folders:
	rm -rf ./data/benchmarks/swingup/
	rm -rf ./data/benchmarks/swimmer
	rm -rf ./data/benchmarks/cylinder
	
	rm -rf ./data/agents/swingup
	rm -rf ./data/agents/swimmer
	rm -rf ./data/agents/cylinder

	rm -rf ./data/hydrogym/cylinder


# THIS COULD BE EXTREMELY DANGEROUS! 
# BE VERY CERTAIN YOU HAVE NOTHING NEW INSIDE THESE FOLDERS WORTH SAVING
clean_data: 
	rm -rf ./data/