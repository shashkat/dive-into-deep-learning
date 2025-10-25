- I set up NVIDIA Nsight and tried to profile my scripts which train the model on 1 gpu and 2 gpus, but there are some issues with running NVIDIA Nsight on containerized envs (modal). 
Error:
FATAL ERROR: /dvs/p4/build/sw/devtools/Agora/Rel/CUDA12.8/QuadD/Target/Daemon/TimeConversion.cpp(312): Throw in function int64_t QuadDDaemon::PostMortemTimeConverter::ConvertGpuTicksToSyncNs(const QuadDCommon::Uuid&, int64_t) const
Dynamic exception type: boost::wrapexcept<QuadDCommon::InternalErrorException>
std::exception::what: InternalErrorException

Hence, skipping this for now.