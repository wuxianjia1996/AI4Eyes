<script setup lang="js">
import {computed, onMounted, reactive, ref, watch} from "vue";
import {ElMessage} from "element-plus";
import * as echarts from "echarts";

const imageUrl = ref("");
const classIndex = ref(-1)
const radio1 = ref('2')
const showSampleFlag = ref(false)
const predictUrl = computed(() => {
  return "http://hw.masaikk.icu:10001/predict" + radio1.value
})
const imgsList = ['http://ai4eye.masaikk.icu/imgs/1.jpg', 'http://ai4eye.masaikk.icu/imgs/2.jpg', 'http://ai4eye.masaikk.icu/imgs/3.jpg', 'http://ai4eye.masaikk.icu/imgs/0.jpg']

const ranksLabelList = computed(() => {
  let classList = []
  switch (radio1.value) {
    case "2": {
      classList = ["normal", "cataract"];

      break
    }
    case "3": {
      classList = ["normal", "moderate", "severe"]

      break
    }
    case "4": {
      classList = ["normal", "mild", "intense", "severe"]

      break
    }
    default: {
      classList = ["normal", "cataract"]
    }
  }
  return classList
})

watch(radio1, (newValue, oldValue) => {
  if (newValue !== oldValue) {
    classMessage.value = "please re-upload"
  }
});


const resultObj = reactive({
  riskList: [],
});

// 创建一个computed属性来合并数据和标签
const mergedData = computed(() => {
  return resultObj.riskList.map((value, index) => {
    return {value: value, name: ranksLabelList.value[index]};
  });
});

/**
 * 用不上
 * @type {UnwrapNestedRefs<{yAxis: {type: string}, xAxis: {data: string[], type: string}, series: [{data: UnwrapRef<*[]>, type: string}], title: {top: string, left: string, text: string}}>}
 */
const option = reactive({
  title: {
    text: "概率分布", // 图表标题
    left: "center", // 标题位置
    top: "50px",
  },
  xAxis: {
    type: "category",
    data: ranksLabelList.value,
  },
  yAxis: {
    type: "value",
    // axisLabel: {
    //   // 在数字后添上百分号
    //   formatter: "{value} %",
    // },
  },
  series: [
    {
      data: resultObj.riskList,
      type: "bar",
    },
  ],
});

const optionPie = reactive({
  title: {
    text: "Probability distribution", // 图表标题
    left: "center", // 标题位置
    top: "10px",
  },
  series: [{
    name: '',
    type: 'pie',
    radius: '50%',
    data: [],
    label: {
      formatter: function (params) {
        return params.name + '\n' + params.percent + '%';
      }
    }
  }]
});

const chartRef = ref(null);
let myChart = null;

onMounted(() => {
      if (chartRef.value) {
        myChart = echarts.init(chartRef.value)
      }
    }
)


const classMessage = computed(() => {
  let classList = ranksLabelList.value
  if (classIndex.value === -1) {
    return ""
  } else {
    return "Diagnosis result: " + classList[classIndex.value]
  }
})

const beforeUpload = (file) => {
  // console.log(file.type);
  const isPNG = ["image/png", "image/jpeg"].includes(file.type);
  if (!isPNG) {
    ElMessage.error("只能上传PNG或者JPG文件");
  }
  return isPNG;
};

const handleUploadSuccess = (response, file) => {
  // 处理上传成功后的返回值
  // console.log("上传成功，返回值为：", response);
  handleUploadPicture(response.probs);
  classIndex.value = response.preds[0]

  const reader = new FileReader();
  reader.onload = (e) => {
    imageUrl.value = e.target?.result
  };
  reader.readAsDataURL(file.raw);
};

const handleUploadFailed = () => {
  ElMessage.error("Network error!");
};


const handleUploadPicture = (riskNumberList) => {
  if (!Array.isArray(riskNumberList)) {
    console.error('输入的riskNumberList不是一个数组');
    return;
  }

  const riskList = riskNumberList.map(item => item * 100);
  resultObj.riskList = riskList
  // console.log('更新后的风险列表:', riskList);

  // if (option && option.series && option.series[0]) {
  //   option.series[0].data = riskList;
  //   option.xAxis.data = ranksLabelList.value;
  //   myChart.setOption(option, true); // 使用true强制更新图表
  // } else {
  //   console.error('图表配置option的格式不正确或series[0]未定义');
  // }

  optionPie.series[0].data = mergedData.value
  console.log(optionPie.series[0].data);
  myChart.setOption(optionPie, true);
};


</script>

<template>
  <div class="main-holder">
    <div class="block-holder">
      <div>
        <h1>AI4Eyes</h1>
      </div>

      <div>
        <el-radio-group v-model="radio1" class="ml-4">
          <el-radio value="2" size="large"><b>2-Cls.</b></el-radio>
          <el-radio value="3" size="large"><b>3-Cls.</b></el-radio>
          <el-radio value="4" size="large"><b>4-Cls.</b></el-radio>
        </el-radio-group>
      </div>

    </div>
    <div>
      <el-upload
          class="upload-demo"
          :action="predictUrl"
          :show-file-list="false"
          list-type="picture-card"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadFailed"
          :before-upload="beforeUpload"
          name="image"
          draggable="true"
      >
        <template v-if="imageUrl">
          <img :src="imageUrl" class="image-preview" alt="Uploaded Image"/>
        </template>
        <template v-else>
          <div>
            <div class="upload-content">
              <i class="el-icon-plus upload-icon"></i>
            </div>
            <div class="upload-text">Upload image</div>
          </div>
        </template>
      </el-upload>
    </div>
    <h3>{{ classMessage }}</h3>
    <div ref="chartRef" style="width: 300px; height: 240px;"></div>

    <div>
      <el-button @click="()=>{showSampleFlag = !showSampleFlag}">Show samples</el-button>
    </div>
    <br>
    <div v-if="showSampleFlag">
      <el-container>
        <el-image v-for="(img, index) in imgsList" :key="index" :src="img"
                  style="margin-right: 10px; margin-bottom: 10px;max-width: 150px;" fit="contain"
                  :preview-src-list="imgsList" ></el-image>
      </el-container>
    </div>

  </div>

</template>

<style scoped>
.image-preview {
  width: 100%; /* 宽度填充整个容器 */
  height: 100%; /* 高度填充整个容器 */
  object-fit: cover; /* 图片保持比例填充，可能会被裁剪 */
}

.block-holder {
  margin: 0;
  display: flex;
  flex-direction: column;
  place-items: center;
}

.main-holder {
  margin: 0;
  display: flex;
  flex-direction: column;
  place-items: center;
}

</style>