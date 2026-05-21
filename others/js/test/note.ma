<template>
  <div>
    <Teleport to="body">
      <div
        :class="{
          'fade-in fixed z-50 top-0 left-0 w-screen h-screen inset-0 bg-gray-400 bg-opacity-30 backdrop-filter backdrop-blur-sm':
            isModalOpen,
        }"
        ref="modalOverlay"
        @click="handleOverlayClick"
      >
        <div
          id="modal"
          ref="modal"
          class="modal bg-white fixed w-full md:w-3/4 max-w-[1024px] max-h-[786px] top-1/2 left-1/2 z-10 translate-x-[-50%] translate-y-[-50%] sm:h-[100dvh] md:h-auto"
          v-show="isModalOpen"
        >
          <button
            class="sq-btn sq-btn-secondary fixed md:absolute md:right-6 md:top-6 right-2 top-2 p-0 rounded-2xl z-100"
            @click="handleModalClose"
            aria-label="close"
          >
            <svg class="!w-8 !h-8" aria-hidden="true">
              <use href="#icon-close"></use>
            </svg>
          </button>
          <div
            v-show="items.length > 1"
            class="flex justify-between w-full absolute top-[calc(20vh_-_16px)] sm:top-[calc(30vh_-_16px)] md:top-[calc(50%_-_16px)]"
          >
            <button
              class="sq-btn sq-btn-secondary p-0 ml-2 md:-ml-20 rounded-2xl z-10"
              :class="isLeftChevronHidden ? 'invisible' : 'visible'"
              @click="handlePrevSlide"
              aria-label="Previous Slide"
            >
              <svg class="!w-8 !h-8" aria-hidden="true">
                <use href="#icon-chevron-left"></use>
              </svg>
            </button>
            <button
              class="sq-btn sq-btn-secondary p-0 mr-2 md:-mr-20 rounded-2xl z-10"
              :class="isRightChevronHidden ? 'invisible' : 'visible'"
              @click="handleNextSlide"
              aria-label="Next Slide"
            >
              <svg class="!w-8 !h-8" aria-hidden="true">
                <use href="#icon-chevron-right"></use>
              </svg>
            </button>
          </div>
          <div tabindex="0" ref="openControl">
            <div class="relative" v-if="item.isVideo">
              <video
                muted
                playsinline
                loop
                controls
                width="100%"
                class="h-[40vh] sm:h-[60vh] md:h-auto object-cover"
                ref="videoElement"
              >
                Your browser does not support the video tag.
                <source :src="item.downloadPath" type="video/mp4" />
              </video>
              <button
                @click="handleVideoClick"
                @keydown.space.prevent="handleVideoClick"
                @keydown.enter.prevent="handleVideoClick"
                type="button"
                class="w-full h-[calc(100%_-_120px)] flex justify-center items-center absolute top-1/2 left-1/2 cursor-pointer translate-x-[-50%] translate-y-[-50%]"
                aria-label="play"
              >
              <div class="w-16 h-16 border-8 border-primary rounded-full bg-white relative -top-3 hover:border-primary-70 focus:border-primary-70" :class="{
                hidden: isPlaying,
              }">
                <svg
                  class="w-12 h-12 fill-primary hover:fill-primary-70 focus:fill-primary-70"
                  aria-hidden="true"
                >
                  <use href="#icon-play"></use>
                </svg>
              </div>
              </button>
            </div>
            <img
              v-else
              :src="item.path"
              :alt="item.description"
              class="w-full md:max-h-[80vh] h-[40vh] sm:h-[60vh] md:h-auto object-cover"
            />
          </div>
          <div
            class="bg-white px-8 py-6 flex justify-end flex-col gap-y-4 md:flex-row"
          >
            <div
              class="flex items-center w-full"
              :class="[disableDownload ? '' : 'md:w-2/3']"
            >
              <div class="line-clamp-3 text-h5">
                {{
                  item.title || item.fileName.split(".").slice(0, -1).join(".")
                }}
              </div>
            </div>
            <div
              v-if="!disableDownload"
              class="flex items-center w-full md:w-1/3 justify-start md:justify-end md:pt-0 md:pr-5"
            >
              <a
                :href="item.downloadPath"
                class="sq-btn sq-btn-sm sq-btn-secondary"
                :download="item.fileName"
              >
                <svg aria-hidden="true">
                  <use href="#icon-download"></use>
                </svg>
                Download {{ "(" + item.downloadSize + ")" }}
              </a>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script>
import { defineComponent } from "vue";

export default defineComponent({
  props: {
    isModalOpen: Boolean,
    disableDownload: Boolean,
    items: Array,
    itemIndex: Number,
  },
  data: () => ({
    isPlaying: false,
    firstFocusableElement: null,
    lastFocusableElement: null,
    openFocus: true,
  }),
  computed: {
    isLeftChevronHidden() {
      return this.itemIndex === 0;
    },
    isRightChevronHidden() {
      return this.itemIndex === this.items.length - 1;
    },
    item() {
      return this.items[this.itemIndex];
    },
  },
  methods: {
    initModal() {
      const modal = this.$refs.modal;
      let focusableElementsString = 'a[href], [tabindex="0"], button';
      this.$nextTick(() => {
        if (modal) {
          let focusableElements = modal.querySelectorAll(
            focusableElementsString,
          );
          focusableElements = Array.prototype.slice.call(focusableElements);
          this.firstFocusableElement = focusableElements[0];
          this.lastFocusableElement =
            focusableElements[focusableElements.length - 1];
        }
        document.addEventListener("keydown", this.trapFocus);
      });
    },
    handleOverlayClick(event) {
      const modal = this.$refs.modal;
      if (!modal.contains(event.target)) {
        this.updateParentValue();
      }
    },
    updateParentValue() {
      // Emit the updated value to the parent component
      this.$emit("updateValue", !this.isModalOpen);
    },
    handleVideoClick(event) {
      const button = event.target.closest("button");
      const video = button.previousElementSibling;
      if (this.isPlaying) {
        video.pause();
      } else {
        video.play();
      }
      this.isPlaying = !this.isPlaying;
    },
    trapFocus(e) {
      if (!this.isModalOpen) return;
      if (!this.openFocus) {
        this.openFocus = true;
        this.firstFocusableElement.classList.remove("focus:bg-secondary");
        this.firstFocusableElement.classList.remove("focus:outline-none");
      }
      const isTabPressed = e.key === "Tab";
      if (!isTabPressed) {
        if (e.key === "Escape") {
          this.updateParentValue();
        }
        return;
      }

      if (e.shiftKey) {
        if (document.activeElement === this.firstFocusableElement) {
          this.lastFocusableElement.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === this.lastFocusableElement) {
          this.firstFocusableElement.focus();
          e.preventDefault();
        }
      }
    },
    handleModalClose() {
      this.updateParentValue();
    },
    handleNextSlide() {
      if (this.itemIndex < this.items.length - 1) {
        if (this.items[this.itemIndex + 1].isVideo) {
          this.isPlaying = false;
          const video = this.$refs.videoElement;
          if (video) {
            video.load();
          }
        }
        this.$emit("update:itemIndex", this.itemIndex + 1);
      }
    },
    handlePrevSlide() {
      if (this.itemIndex > 0) {
        if (this.items[this.itemIndex - 1].isVideo) {
          this.isPlaying = false;
          const video = this.$refs.videoElement;
          if (video) {
            video.load();
          }
        }
        this.$emit("update:itemIndex", this.itemIndex - 1);
      }
    },
  },
  watch: {
    isModalOpen(isOpen) {
      const el = this.$refs.modalOverlay;
      if (isOpen) {
        const video = this.$refs.videoElement;
        if (this.item.isVideo) {
          if (video) {
            video.load();
            this.isPlaying = false;
          }
        }
        document.body.style.overflow = "hidden";
        this.$nextTick(() => {
          if (this.firstFocusableElement) {
            this.firstFocusableElement.focus();
            if (this.openFocus) {
              this.openFocus = false;
              this.firstFocusableElement.classList.add("focus:bg-secondary");
              this.firstFocusableElement.classList.add("focus:outline-none");
            }
          }
        });
      } else {
        if (el) {
          document.body.style.overflow = "auto";
        }
        this.$nextTick(() => {
          if (this.$refs.openControl) {
            this.$refs.openControl.focus();
          }
        });
      }
    }
  },
  mounted() {
    this.initModal();
  },
  beforeDestroy() {
    document.removeEventListener("keydown", this.trapFocus);
  },
});
</script>

<style scoped>
.modal {
  animation-duration: 0.6s;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
.fade-in {
  animation-name: fadeIn;
}
</style>
