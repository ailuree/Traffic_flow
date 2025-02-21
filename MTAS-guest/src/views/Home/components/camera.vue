<template>
    <div class="">

        <!-- 拍照 -->
        <div class="camera-card">
            <div>
                <!-- 摄像头调用 -->
                <video ref="videoCamera" :width="videoWidth" :height="videoHeight" autoplay></video>

                <!-- 用绘制canvas-转换给img -->
                <canvas style="display: none" ref="canvasCamera" :width="videoWidth" :height="videoHeight"></canvas>
            </div>
        </div>

        <!-- 3个按钮 -->
        <div class="card-child">
            <el-button @click="getCompetence()" type="primary" class="btn_photo">开启</el-button>
            <el-button @click="stopNavigator()" type="warning" class="btn_photo">关闭</el-button>
            <el-button @click="setImage()" type="success" class="btn_photo">拍照</el-button>
        </div>


        <!-- 拍照结果 -->
        <div v-if="imgSrc" class="img_res_camera">
            <div class="card-title">
                拍摄预览
            </div>
            <img :src="imgSrc" alt class="res_img" />
        </div>

        <!-- 上传按钮 -->
        <div class="card-child">
            <el-button v-if="imgSrc" type="danger" class="btn_photo">取消</el-button>
            <el-button v-if="imgSrc" type="warning" class="btn_photo">分享</el-button>
            <el-button v-if="imgSrc" type="success" @click="sendPhotoToServer(this.imgSrc)" class="btn_photo">上传</el-button>
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from 'vue';
import axios from 'axios'; // 导入axios

export default defineComponent({
    data() {
        return {
            videoWidth: 300,
            videoHeight: 200,
            imgSrc: "",
            thisCanvas: null as HTMLCanvasElement | null,
            thisContext: null as CanvasRenderingContext2D | null,
            thisVideo: null as HTMLVideoElement | null,
            openVideo: false,
            info: "",
        };
    },
    mounted() {
        this.getCompetence(); // 进入页面就调用摄像头

        this.updateVideoSize(); // 初始化时调用一次

        // 监听窗口大小变化，动态更新 videoWidth 和 videoHeight
        window.addEventListener('resize', this.updateVideoSize);
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.updateVideoSize);
    },
    methods: {
        // 更新video大小
        updateVideoSize() {
            const screenWidth = window.innerWidth;
            if (screenWidth < 600) {
                this.videoWidth = 200;
                this.videoHeight = 130;
                return
            }
            this.videoWidth = 300;
            this.videoHeight = 200;
            console.log(this.videoWidth)
        },

        // 上传照片
        sendPhotoToServer(photoData: string) {
            const url = 'http://127.0.0.1:5500/photo'; // 后端接口地址
            console.log(photoData);

            // 构造请求参数
            const params = new FormData();
            params.append('photo', photoData);

            // 使用 axios 发送 POST 请求
            axios.post(url, params)
                .then(response => {
                    console.log('照片上传成功:', response);
                    this.info = response
                    // 处理后端返回的响应
                    this.$emit('message', response.data); // 触发自定义事件并传递信息
                })
                .catch(error => {
                    // 处理后端返回的响应
                    this.$emit('message', response.data); // 触发自定义事件并传递信息
                    console.error('照片上传失败', error);

                });
        },

        // 调用权限（打开摄像头功能）
        getCompetence() {
            const thisCanvas = this.$refs.canvasCamera as HTMLCanvasElement;
            const thisContext = thisCanvas.getContext("2d");
            const thisVideo = this.$refs.videoCamera as HTMLVideoElement;
            this.thisCanvas = thisCanvas;
            this.thisContext = thisContext;
            this.thisVideo = thisVideo;
            this.thisVideo.style.display = "block";
            // 获取媒体属性，旧版本浏览器可能不支持 mediaDevices，我们首先设置一个空对象
            if (navigator.mediaDevices === undefined) {
                navigator.mediaDevices = {} as any;
            }
            // 一些浏览器实现了部分 mediaDevices，我们不能只分配一个对象
            // 使用 getUserMedia，因为它会覆盖现有的属性。
            // 这里，如果缺少 getUserMedia 属性，就添加它。
            if (navigator.mediaDevices.getUserMedia === undefined) {
                navigator.mediaDevices.getUserMedia = function (
                    constraints: MediaStreamConstraints
                ) {
                    // 首先获取现存的 getUserMedia（如果存在）
                    const getUserMedia =
                        navigator.webkitGetUserMedia ||
                        navigator.mozGetUserMedia ||
                        navigator.getUserMedia;
                    // 有些浏览器不支持，会返回错误信息
                    // 保持接口一致
                    if (!getUserMedia) {
                        // 不存在则报错
                        return Promise.reject(
                            new Error("getUserMedia is not implemented in this browser")
                        );
                    }
                    // 否则，使用 Promise 将调用包装到旧的 navigator.getUserMedia
                    return new Promise(function (resolve, reject) {
                        getUserMedia.call(navigator, constraints, resolve, reject);
                    });
                };
            }
            const constraints: MediaStreamConstraints = {
                audio: false,
                video: {
                    width: this.videoWidth,
                    height: this.videoHeight,
                    // transform: "scaleX(-1)",
                },
            };
            navigator.mediaDevices
                .getUserMedia(constraints)
                .then((stream) => {
                    // 旧的浏览器可能没有 srcObject
                    if ("srcObject" in this.thisVideo!) {
                        (this.thisVideo! as any).srcObject = stream;
                    } else {
                        // 避免在新的浏览器中使用它，因为它正在被弃用。
                        (this.thisVideo! as any).src = window.URL.createObjectURL(stream);
                    }
                    this.thisVideo!.onloadedmetadata = function (e) {
                        (this! as any).play();
                    };
                })
                .catch((err) => {
                    console.log(err);
                });
        },
        // 绘制图片（拍照功能）
        setImage() {
            // canvas 画图
            this.thisContext!.drawImage(
                this.thisVideo!,
                0,
                0,
                this.videoWidth,
                this.videoHeight
            );
            // 获取图片 base64 链接
            const image = this.thisCanvas!.toDataURL("image/png");
            this.imgSrc = image; // 赋值并预览图片
        },
        // 关闭摄像头
        stopNavigator() {
            (this.thisVideo!.srcObject as MediaStream).getTracks()[0].stop();
        },
    },
});
</script>



<style scoped>
.content {
    display: flex;
    flex-direction: column;
    width: 80%;
    /* height: 80vh; */
    margin: auto;
    margin-top: 10vh;
    margin-bottom: 10vh;
    user-select: none;
}


.card-child {
    padding: 0.5rem;
}


@media (max-width: 600px) {
    .btn_photo {
        width: 3rem;
        /* font: 1rem sans-serif; */
        font-size: 1vw;
    }
}

.camera-card {
    display: flex;
    justify-content: flex-start;
    align-items: center;
}

.card-title {
    font-size: 1.3vw;
    font-weight: bold;
}
</style>



