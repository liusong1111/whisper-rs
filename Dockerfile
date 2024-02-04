ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.7.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
WORKDIR /app

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all
# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable cuBLAS
ENV WHISPER_CUBLAS=1

RUN apt-get update
RUN apt-get install -y ffmpeg build-essential gcc cmake vim tmux wget curl net-tools 

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update
RUN apt-get install -y clang-17 lldb-17 lld-17

# RUN apt-get install -y libllvm-17-ocaml-dev libllvm17 llvm-17 llvm-17-dev llvm-17-doc llvm-17-examples llvm-17-runtime
# RUN apt-get install -y clang-17 clang-tools-17 clang-17-doc libclang-common-17-dev libclang-17-dev libclang1-17 clang-format-17 python3-clang-17 clangd-17 clang-tidy-17
# RUN apt-get install -y lldb-17
# RUN apt-get install -y lld-17
# RUN apt-get install -y libc++-17-dev libc++abi-17-dev
# RUN apt-get install -y libomp-17-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
COPY cargo_config /root/.cargo/config
ENV PATH=/root/.cargo/bin:$PATH

RUN cargo install gitui

# Ref: https://stackoverflow.com/a/53464012
ENV CUDA_MAIN_VERSION=11.7
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA_MAIN_VERSION}/compat:$LD_LIBRARY_PATH

COPY . .
RUN cargo build --release --examples --features cuda

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime
WORKDIR /app

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all
# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable cuBLAS
ENV WHISPER_CUBLAS=1

RUN apt-get update
RUN apt-get install -y ffmpeg build-essential gcc cmake vim tmux wget curl net-tools 

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update
RUN apt-get install -y clang-17 lldb-17 lld-17
# RUN apt-get install -y libllvm-17-ocaml-dev libllvm17 llvm-17 llvm-17-dev llvm-17-doc llvm-17-examples llvm-17-runtime
# RUN apt-get install -y clang-17 clang-tools-17 clang-17-doc libclang-common-17-dev libclang-17-dev libclang1-17 clang-format-17 python3-clang-17 clangd-17 clang-tidy-17
# RUN apt-get install -y lldb-17
# RUN apt-get install -y lld-17
# RUN apt-get install -y libc++-17-dev libc++abi-17-dev
# RUN apt-get install -y libomp-17-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
COPY cargo_config /root/.cargo/config
ENV PATH=/root/.cargo/bin:$PATH

RUN cargo install gitui

# Ref: https://stackoverflow.com/a/53464012
ENV CUDA_MAIN_VERSION=11.7
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA_MAIN_VERSION}/compat:$LD_LIBRARY_PATH

COPY --from=build /app /app

ENTRYPOINT [ "bash", "-c" ]
