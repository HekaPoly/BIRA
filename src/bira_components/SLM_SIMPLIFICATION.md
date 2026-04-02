# SLM_Manager Quick Reference

## Build the model

```bash
cd /Users/home/Projets/BIRA
ollama create bira-assistant -f Modelfile
```

## Run BIRA

```bash
cd /Users/home/Projets/BIRA/src
python main.py --mock --SLM_DEBUG // SLM_DEBUG to see details from how the SLM ats
```

## Optional checks

```bash
ollama list | grep bira
ollama run bira-assistant "give me a bottle"
```
