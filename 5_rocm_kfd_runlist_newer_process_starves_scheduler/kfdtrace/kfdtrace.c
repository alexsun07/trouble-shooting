// LD_PRELOAD shim: report who touches the GPU. On the first open("/dev/kfd") and on each KFD
// CREATE_QUEUE / first few ALLOC_MEMORY_OF_GPU ioctls, append a marker to $KFDTRACE_DIR/trace.<pid>.txt
// and raise $KFDTRACE_SIG (default 40) so Python's faulthandler (registered by sitecustomize.py) dumps
// every thread's Python stack into the same file. Never raises before a handler is installed.
//   gcc -O2 -shared -fPIC -o kfdtrace.so kfdtrace.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int n_queue, n_alloc;

static void report(const char *what, unsigned long arg) {
  const char *dir = getenv("KFDTRACE_DIR");
  if (!dir) return;
  char path[512], buf[256];
  snprintf(path, sizeof path, "%s/trace.%d.txt", dir, getpid());
  int fd = ((int (*)(const char *, int, ...))dlsym(RTLD_NEXT, "open"))(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) return;
  char comm[64] = "";
  FILE *f = fopen("/proc/self/comm", "r");
  if (f) { if (fgets(comm, sizeof comm, f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
  int n = snprintf(buf, sizeof buf, "\n=== KFDTRACE pid=%d tid=%d comm=%s event=%s arg=%lu\n", getpid(), gettid(), comm, what, arg);
  write(fd, buf, n);
  close(fd);
  int sig = getenv("KFDTRACE_SIG") ? atoi(getenv("KFDTRACE_SIG")) : 40;
  struct sigaction old;
  if (sigaction(sig, NULL, &old) == 0 && old.sa_handler != SIG_DFL && old.sa_handler != SIG_IGN) raise(sig);
}

static int is_kfd(const char *p) { return p && strcmp(p, "/dev/kfd") == 0; }

int open(const char *p, int flags, ...) {
  static int (*real)(const char *, int, ...);
  if (!real) real = dlsym(RTLD_NEXT, "open");
  va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
  int r = real(p, flags, mode);
  if (is_kfd(p)) report("open_kfd", 0);
  return r;
}
int open64(const char *p, int flags, ...) {
  static int (*real)(const char *, int, ...);
  if (!real) real = dlsym(RTLD_NEXT, "open64");
  va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
  int r = real(p, flags, mode);
  if (is_kfd(p)) report("open_kfd", 0);
  return r;
}
int openat(int d, const char *p, int flags, ...) {
  static int (*real)(int, const char *, int, ...);
  if (!real) real = dlsym(RTLD_NEXT, "openat");
  va_list ap; va_start(ap, flags); int mode = va_arg(ap, int); va_end(ap);
  int r = real(d, p, flags, mode);
  if (is_kfd(p)) report("open_kfd", 0);
  return r;
}
int ioctl(int fd, unsigned long req, ...) {
  static int (*real)(int, unsigned long, ...);
  if (!real) real = dlsym(RTLD_NEXT, "ioctl");
  va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
  if (_IOC_TYPE(req) == 'K') {
    unsigned nr = _IOC_NR(req);
    if (nr == 0x02 && n_queue < 32) report("create_queue", ++n_queue);           // AMDKFD_IOC_CREATE_QUEUE
    if (nr == 0x16 && n_alloc < 3) report("alloc_memory_of_gpu", ++n_alloc);    // AMDKFD_IOC_ALLOC_MEMORY_OF_GPU
  }
  return real(fd, req, arg);
}
