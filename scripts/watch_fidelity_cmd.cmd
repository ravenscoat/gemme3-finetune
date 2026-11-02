@echo off
title Gemma - Fidelity experiments
cd /d D:\gemma-finetune
if exist D:\gemma-finetune\reports\fidelity_alignment_status.json (
D:\gemma-finetune\.venv-windows\Scripts\python.exe -u D:\gemma-finetune\scripts\watch_alignment_cmd.py
) else (
D:\gemma-finetune\.venv-windows\Scripts\python.exe -u D:\gemma-finetune\scripts\watch_fidelity_cmd.py
)
