**1、移除错误缓存**
 首先应该移除所有错误的 cache，对于文件：



git rm --cached path_of_a_giant_file

| 1    | git rm --cached path_of_a_giant_file |
| ---- | ------------------------------------ |
|      |                                      |



对于文件夹：



git rm --cached -r path_of_a_giant_dir

| 1    | git rm --cached -r path_of_a_giant_dir |
| ---- | -------------------------------------- |
|      |                                        |



例如对于我的例子就是这样的：



git rm --cached -r Examples/iOSDemo/Pods/dependency/libg2o.a

| 1    | git rm --cached -r Examples/iOSDemo/Pods/dependency/libg2o.a |
| ---- | ------------------------------------------------------------ |
|      |                                                              |



**2、重新提交：**
 编辑最后提交信息：



git commit --amend

| 1    | git commit --amend |
| ---- | ------------------ |
|      |                    |



修改 log 信息后保存返回。

重新提交：



git push

| 1    | git push |
| ---- | -------- |
|      |          |



PS：如果上面的步骤仍然无法解决问题，则可以运行如下命令删除有关某个文件的push操作：



git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch YOUR-FILE'

| 1    | git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch YOUR-FILE' |
| ---- | ------------------------------------------------------------ |
|      |                                                              |